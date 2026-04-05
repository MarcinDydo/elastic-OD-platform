import logging
import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted
from dask_ml.feature_extraction.text import CountVectorizer
import dask.array as da
from dask_ml.decomposition import TruncatedSVD
import dask.dataframe as dd
from pyod.models import *
from pyod.models.base import BaseDetector

logger = logging.getLogger(__name__)

class TfidfVectorizerWrapper(TfidfVectorizer):
    """TfidfVectorizer subclass that materialises dask inputs.

    dask_ml does not provide a TfidfVectorizer, so we subclass sklearn's
    implementation and call ``.compute()`` on any dask collection before
    delegating to the parent class.  All ``__init__`` parameters,
    ``get_params``, and ``set_params`` are inherited automatically.
    """

    @staticmethod
    def _ensure_concrete(X):
        if hasattr(X, "compute"):
            return X.compute()
        return X

    @staticmethod
    def _densify(X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.asarray(X, dtype="float64")

    def fit(self, raw_documents, y=None):
        return super().fit(self._ensure_concrete(raw_documents), y)

    def transform(self, raw_documents):
        return self._densify(super().transform(self._ensure_concrete(raw_documents)))

    def fit_transform(self, raw_documents, y=None):
        return self._densify(super().fit_transform(self._ensure_concrete(raw_documents), y))

class CountVectorizerWrapper(BaseEstimator, TransformerMixin):
    """wrapper around dask_ml CountVectorizer.
    """

    _dask_native = True

    def __init__(self, **kwargs):
        self._cv_params = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_params(self, deep=True):
        return dict(self._cv_params)

    def set_params(self, **params):
        self._cv_params.update(params)
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @staticmethod
    def _densify_dask(X):
        if isinstance(X, da.Array):
            return X.map_blocks(
                lambda block: block.toarray() if hasattr(block, "toarray") else block,
                dtype="float64",
            )
        if hasattr(X, "toarray"):
            return X.toarray()
        return X

    def fit(self, raw_documents, y=None):
        self._model = CountVectorizer(**self._cv_params)
        self._model.fit(raw_documents)
        return self

    def transform(self, raw_documents):
        return self._densify_dask(self._model.transform(raw_documents))

    def fit_transform(self, raw_documents, y=None):
        self._model = CountVectorizer(**self._cv_params)
        return self._densify_dask(self._model.fit_transform(raw_documents, y))


class PyODDetectorWrapper(BaseEstimator):
    """Generic wrapper for any PyOD BaseDetector subclass.

    Bridges PyOD to the sklearn Pipeline/GridSearchCV interface:
    - Materializes dask arrays to numpy (block-by-block to limit peak memory)
    - Converts PyOD labels (0=inlier, 1=outlier) to sklearn convention (1=inlier, -1=outlier)
    - Exposes labels_ and decision_scores_ after fit
    - PyThresh objects in contamination flow through to PyOD natively
    """

    def __init__(self, pyod_cls=None, **kwargs):
        self.pyod_cls = pyod_cls
        self._init_kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_params(self, deep=True):
        params = {"pyod_cls": self.pyod_cls}
        params.update(self._init_kwargs)
        return params

    def set_params(self, **params):
        if "pyod_cls" in params:
            self.pyod_cls = params.pop("pyod_cls")
        self._init_kwargs.update(params)
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @staticmethod
    def _to_numpy(X):
        """Materialize dask/sparse to dense numpy, block-by-block for dask."""
        if isinstance(X, da.Array):
            chunks = []
            for i in range(X.numblocks[0]):
                block = X.blocks[i].compute()
                if hasattr(block, "toarray"):
                    block = block.toarray()
                chunks.append(np.asarray(block, dtype="float64"))
            arr = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
        elif hasattr(X, "compute"):
            arr = np.asarray(X.compute(), dtype="float64")
        elif hasattr(X, "toarray"):
            arr = np.asarray(X.toarray(), dtype="float64")
        else:
            arr = np.asarray(X, dtype="float64")
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_model(self):
        return self.pyod_cls(**self._init_kwargs)

    def fit(self, X, y=None):
        arr = self._to_numpy(X)
        self._model = self._build_model()
        self._model.fit(arr)
        self.labels_ = np.where(self._model.labels_ == 1, -1, 1)
        self.decision_scores_ = self._model.decision_scores_
        return self

    def predict(self, X):
        arr = self._to_numpy(X)
        raw = self._model.predict(arr)
        return np.where(raw == 1, -1, 1)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

    def decision_function(self, X):
        arr = self._to_numpy(X)
        return self._model.decision_function(arr)
    
class AutoencoderWrapper(BaseEstimator, TransformerMixin):
    """Autoencoder-based dimensionality reduction (feature transformer).

    Builds a symmetric encoder-decoder from ``hidden_neurons`` and trains
    on reconstruction loss.  After fit the encoder output (last layer of
    ``hidden_neurons``) is used as the reduced representation.

    Config example::

        "params": {"hidden_neurons": [1000, 500, 50], "epochs": 100}

    Encoder: input_dim → 1000 → 500 → 50  (bottleneck = 50-d output)
    Decoder: 50 → 500 → 1000 → input_dim
    """

    def __init__(self, hidden_neurons=None, epochs=20, lr=1e-3, batch_size=256):
        self.hidden_neurons = hidden_neurons or [128, 64]
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    @staticmethod
    def _to_numpy(X):
        if hasattr(X, "compute"):
            X = X.compute()
        if hasattr(X, "toarray"):
            X = X.toarray()
        arr = np.asarray(X, dtype="float32")
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X, y=None):
        import torch
        from torch import nn

        arr = self._to_numpy(X)
        input_dim = arr.shape[1]

        # Build encoder: input_dim → hidden_neurons[0] → … → hidden_neurons[-1]
        encoder_layers = []
        prev = input_dim
        for units in self.hidden_neurons:
            encoder_layers += [nn.Linear(prev, units), nn.ReLU(), nn.BatchNorm1d(units)]
            prev = units
        self.encoder_ = nn.Sequential(*encoder_layers)

        # Build decoder: mirror of encoder back to input_dim
        decoder_layers = []
        for units in reversed(self.hidden_neurons[:-1]):
            decoder_layers += [nn.Linear(prev, units), nn.ReLU(), nn.BatchNorm1d(units)]
            prev = units
        decoder_layers.append(nn.Linear(prev, input_dim))
        decoder = nn.Sequential(*decoder_layers)

        autoencoder = nn.Sequential(self.encoder_, decoder)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(torch.tensor(arr))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
        )

        autoencoder.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                loss = criterion(autoencoder(batch), batch)
                loss.backward()
                optimizer.step()

        self.encoder_.eval()
        return self

    def transform(self, X, y=None):
        import torch
        arr = self._to_numpy(X)
        with torch.no_grad():
            encoded = self.encoder_(torch.tensor(arr)).numpy()
        return encoded.astype("float64")

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class TruncatedSVDWrapper(BaseEstimator, TransformerMixin):
    """wrapper around dask_ml TruncatedSVD.
    Accepts sparse, dense, or dask input and always returns a dask array
    with reduced dimensionality.  Converts numpy/sparse input to a
    single-chunk dask array since ``da.linalg.svd`` requires
    ``numblocks[1] == 1``.

    """

    _dask_native = True

    def __init__(self, n_components=2, algorithm="randomized"):
        self.n_components = n_components
        self.algorithm = algorithm

    @staticmethod
    def _to_dask(X):
        if isinstance(X, da.Array):
            if any(np.isnan(s) for c in X.chunks for s in c):
                X = X.compute_chunk_sizes()
            return X
        if hasattr(X, "toarray"):
            X = X.toarray()
        arr = np.asarray(X, dtype="float64")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return da.from_array(arr, chunks=arr.shape)

    def fit(self, X, y=None):
        self._model = TruncatedSVD(n_components=self.n_components, algorithm=self.algorithm)
        self._model.fit(self._to_dask(X))
        return self

    def transform(self, X, y=None):
        return self._model.transform(self._to_dask(X))

    def fit_transform(self, X, y=None):
        self._model = TruncatedSVD(n_components=self.n_components, algorithm=self.algorithm)
        return self._model.fit_transform(self._to_dask(X))

class PositionalIDFWrapper(BaseEstimator, TransformerMixin):
    """tokenises each
    text value, computes corpus-wide IDF via *TfidfVectorizer*, then writes
    the IDF weight of the token at position *i* into column *i*
    (zero-padded when there are fewer tokens).
    """

    def __init__(
        self,
        token_pattern: str = r"[^\s/\\]+",
        normalize: bool = False,
        max_features: int = 20,
        min_df: int = 2,
        max_df: float = 0.5,
    ):
        self.token_pattern = token_pattern
        self.normalize = normalize
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, X, y=None):
        if hasattr(X, "compute"):
            X = X.compute()
        series = pd.Series(X).fillna("").astype(str)
        vec = TfidfVectorizer(
            token_pattern=self.token_pattern,
            use_idf=True,
            smooth_idf=True,
            norm=None,
            max_features=None,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        vec.fit(series)
        self.idf_map_ = dict(zip(vec.get_feature_names_out(), vec.idf_))
        return self

    def transform(self, X, y=None):
        if hasattr(X, "compute"):
            X = X.compute()
        pat = re.compile(self.token_pattern)
        n = self.max_features
        rows = []
        for text in pd.Series(X).fillna("").astype(str):
            tokens = pat.findall(text)
            row = [
                self.idf_map_.get(tokens[i], 0.0) if i < len(tokens) else 0.0
                for i in range(n)
            ]
            rows.append(row)
        arr = np.array(rows, dtype="float64")
        if self.normalize:
            col_min = arr.min(axis=0)
            col_max = arr.max(axis=0)
            span = col_max - col_min
            span[span == 0] = 1.0
            arr = (arr - col_min) / span
        return arr


class HDBSCAN(BaseDetector):
    """Local reimplementation of PyOD's HDBSCAN detector.

    PyOD removed ``pyod.models.hdbscan`` so we re-implement it here as a
    ``BaseDetector`` subclass wrapping ``sklearn.cluster.HDBSCAN``.  Outlier
    score is ``1 - probabilities_`` (noise points get 1.0).  For scoring
    unseen points we use weighted KNN interpolation over the training
    scores.
    """

    def __init__(self, min_cluster_size=5, min_samples=None,
                 metric='euclidean', alpha=1.0, algorithm='auto',
                 leaf_size=40, n_jobs=1, contamination=0.1):
        super(HDBSCAN, self).__init__(contamination=contamination)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.alpha = alpha
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = check_array(X)
        self._set_n_classes(y)

        try:
            from sklearn.cluster import HDBSCAN as sklearn_HDBSCAN
        except Exception as e:
            raise ImportError(
                "HDBSCAN requires scikit-learn with sklearn.cluster.HDBSCAN. "
                "Please upgrade scikit-learn."
            ) from e

        self.detector_ = sklearn_HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            alpha=self.alpha,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            store_centers='centroid',
        )
        self.detector_.fit(X)

        self.cluster_labels_ = self.detector_.labels_
        self.decision_scores_ = 1.0 - self.detector_.probabilities_
        self._process_decision_scores()

        self.X_train_ = X
        self.tree_ = NearestNeighbors(
            n_neighbors=min(self.min_cluster_size, X.shape[0]),
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        self.tree_.fit(X)

        return self

    def decision_function(self, X):
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        X = check_array(X)

        dist, ind = self.tree_.kneighbors(X)
        weights = 1.0 / (dist + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)

        neighbor_scores = self.decision_scores_[ind]
        scores = np.sum(weights * neighbor_scores, axis=1)
        return scores.ravel()
