import logging
import math
import re
from collections import Counter
from contextlib import contextmanager
import pandas as pd
import numpy as np
from joblib import parallel_backend
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher as _SklearnFeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted
import dask.array as da
import dask.dataframe as dd
from pyod.models import *
from pyod.models.base import BaseDetector

logger = logging.getLogger(__name__)


@contextmanager
def _dask_joblib():
    """Activate joblib's dask backend if a distributed client is running.

    retrieves the current active Client within a Dask worker task, allowing workers to submit new tasks, scatter, or gather results
    """
    try:
        from dask.distributed import get_client
        get_client()
    except (ImportError, ValueError):
        yield
        return
    with parallel_backend("dask"):
        yield


def _gpu_backend() -> bool:
    """True when the active dask client runs on a CUDA cluster."""
    try:
        from dask.distributed import get_client
        client = get_client()
    except (ImportError, ValueError):
        return False
    cluster_cls = type(client.cluster).__name__ if client.cluster is not None else ""
    if "CUDA" in cluster_cls:
        return True
    try:
        info = client.scheduler_info()
        for w in info.get("workers", {}).values():
            if w.get("resources", {}).get("GPU", 0) > 0:
                return True
    except Exception:
        pass
    return False


class TfidfTransformerWrapper(BaseEstimator, TransformerMixin):
    """IDF re-weighting for a pre-computed count matrix.

    Expects input produced by ``dask_ml.feature_extraction.text.CountVectorizer``
    (declared as a separate step in the feature pipeline).  GPU path
    (DASK_DATAFRAME__BACKEND=cudf) delegates to ``cuml.dask`` TfidfTransformer
    so IDF fitting and weighting stay on workers; CPU path fits sklearn's
    ``TfidfTransformer`` once, then applies it per-block via ``map_blocks``
    under the dask joblib backend so transform parallelism is preserved.
    """

    def __init__(self, norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def _sk_kwargs(self):
        norm = self.norm
        if isinstance(norm, str) and norm.lower() == "none":
            norm = None
        return {
            "norm": norm,
            "use_idf": self.use_idf,
            "smooth_idf": self.smooth_idf,
            "sublinear_tf": self.sublinear_tf,
        }

    def _fit_gpu(self, X):
        from cuml.dask.feature_extraction.text import TfidfTransformer as CumlTfidfTransformer
        self._impl_ = CumlTfidfTransformer()
        self._impl_.fit(X)
        return self

    def _fit_cpu(self, X):
        self._impl_ = TfidfTransformer(**self._sk_kwargs())
        if isinstance(X, da.Array):
            X_concrete = X.compute()
        elif hasattr(X, "compute"):
            X_concrete = X.compute()
        else:
            X_concrete = X
        with _dask_joblib():
            self._impl_.fit(X_concrete)
        return self

    def _transform_cpu(self, X):
        impl = self._impl_

        def _block(b):
            out = impl.transform(b)
            return out.toarray() if hasattr(out, "toarray") else np.asarray(out)

        if isinstance(X, da.Array):
            with _dask_joblib():
                return X.map_blocks(_block, dtype="float64")
        with _dask_joblib():
            return _block(X)

    def fit(self, X, y=None):
        return self._fit_gpu(X) if _gpu_backend() else self._fit_cpu(X)

    def transform(self, X):
        if _gpu_backend():
            return self._impl_.transform(X)
        return self._transform_cpu(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)



class AutoencoderWrapper(BaseEstimator, TransformerMixin):
    """Autoencoder-based dimensionality reduction (feature transformer).

    Builds a symmetric encoder-decoder from ``hidden_neurons`` and trains
    on reconstruction loss.  After fit the encoder output (last layer of
    ``hidden_neurons``) is used as the reduced representation.

    Config example::
        "params": {"hidden_neurons": [1000, 500, 50], "epochs": 100}
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

        # Build encoder
        encoder_layers = []
        prev = input_dim
        for units in self.hidden_neurons:
            encoder_layers += [nn.Linear(prev, units), nn.ReLU(), nn.BatchNorm1d(units)]
            prev = units
        self.encoder_ = nn.Sequential(*encoder_layers)

        # Build decoder
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


class RatiosWrapper(BaseEstimator, TransformerMixin):
    """Computes string-analysis feature ratios for each sample.

    Produces a 5-column vector per sample:
      0 alphanumeric character ratio
      1 special character ratio (non-alnum, non-whitespace)
      2 illegal special character ratio (<, >, |, {, }, etc.)
      3 URL-encoded character ratio (%XX sequences)
      4 Shannon entropy of the character distribution
    """

    _ILLEGAL_CHARS = frozenset('<>|{}~^`[]')
    _ENCODED_RE = re.compile(r'%[0-9a-fA-F]{2}')

    def __init__(self, lowercase=False):
        self.lowercase = lowercase

    _COLUMNS = ("alnum", "special", "illegal", "encoded", "entropy")

    def fit(self, X, y=None):
        return self

    @classmethod
    def _rows_for_series(cls, series: pd.Series, lowercase: bool) -> np.ndarray:
        illegal = cls._ILLEGAL_CHARS
        encoded_pat = cls._ENCODED_RE
        rows = []
        for text in series.fillna("").astype(str):
            s = text.lower() if lowercase else text
            n = len(s)
            if n == 0:
                rows.append([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            alnum = sum(1 for c in s if c.isalnum())
            special = sum(1 for c in s if not c.isalnum() and not c.isspace())
            illegal_count = sum(1 for c in s if c in illegal)
            encoded = len(encoded_pat.findall(s)) * 3
            freq = Counter(s)
            entropy = -sum(
                (cnt / n) * math.log2(cnt / n) for cnt in freq.values()
            )
            rows.append([
                alnum / n, special / n, illegal_count / n, encoded / n, entropy,
            ])
        return np.array(rows, dtype="float64")

    def transform(self, X, y=None):
        if isinstance(X, dd.Series):
            cols = self._COLUMNS
            lowercase = self.lowercase
            meta = pd.DataFrame({c: pd.Series([], dtype="float64") for c in cols})
            def _part(s):
                return pd.DataFrame(
                    self._rows_for_series(s, lowercase),
                    columns=cols, index=s.index,
                )
            with _dask_joblib():
                result_dd = X.map_partitions(_part, meta=meta)
                return result_dd.to_dask_array(lengths=True)
        if hasattr(X, "compute"):
            X = X.compute()
        return self._rows_for_series(pd.Series(X), self.lowercase)


class DocumentPoolWrapper(BaseEstimator, TransformerMixin):
    """Pretrained FastText embeddings loaded from a .bin file.

    Token vectors are looked up from the loaded model and averaged into
    a single document vector per sample.
    """

    def __init__(self, model="data/model.bin", pattern=r"[\w.\W]"):
        self.model = model
        self.pattern = pattern

    def fit(self, X, y=None):
        from gensim.models.fasttext import load_facebook_model

        self.ft_model_ = load_facebook_model(self.model)
        self._pat = re.compile(self.pattern)
        return self

    @staticmethod
    def _rows_for_series(series: pd.Series, wv, pat, dim: int) -> np.ndarray:
        rows = []
        for text in series.fillna("").astype(str):
            tokens = pat.findall(text)
            if not tokens:
                rows.append(np.zeros(dim, dtype="float64"))
                continue
            vecs = np.array([wv[t] for t in tokens])
            rows.append(vecs.mean(axis=0))
        return np.array(rows, dtype="float64")

    def transform(self, X, y=None):
        check_is_fitted(self, ["ft_model_"])
        wv = self.ft_model_.wv
        dim = wv.vector_size
        pat = self._pat

        if isinstance(X, dd.Series):
            cols = [f"v{i}" for i in range(dim)]
            meta = pd.DataFrame({c: pd.Series([], dtype="float64") for c in cols})
            def _part(s):
                return pd.DataFrame(
                    self._rows_for_series(s, wv, pat, dim),
                    columns=cols, index=s.index,
                )
            with _dask_joblib():
                result_dd = X.map_partitions(_part, meta=meta)
                return result_dd.to_dask_array(lengths=True)
        if hasattr(X, "compute"):
            X = X.compute()
        return self._rows_for_series(pd.Series(X), wv, pat, dim)


class PyODDetectorWrapper(BaseEstimator):
    """Generic wrapper for any PyOD BaseDetector subclass.

    Bridges PyOD to the sklearn Pipeline/GridSearchCV interface:
    - Materializes dask arrays to numpy (block-by-block to limit peak memory)
    - Converts PyOD labels (0=inlier, 1=outlier) to sklearn convention (1=inlier, -1=outlier)
    - Exposes labels_ and decision_scores_ after fit
    - PyThresh objects in contamination 
    """

    def __init__(self, pyod_class=None, **kwargs):
        self.pyod_class = pyod_class
        self._init_kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_params(self, deep=True):
        params = {"pyod_class": self.pyod_class}
        params.update(self._init_kwargs)
        return params

    def set_params(self, **params):
        if "pyod_class" in params:
            self.pyod_class = params.pop("pyod_class")
        self._init_kwargs.update(params)
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @staticmethod
    def _to_numpy(X):
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
        return self.pyod_class(**self._init_kwargs)

    def fit(self, X, y=None):
        arr = self._to_numpy(X)
        self._model = self._build_model()
        with _dask_joblib():
            self._model.fit(arr)
        self.labels_ = np.where(self._model.labels_ == 1, -1, 1)
        self.decision_scores_ = self._model.decision_scores_
        return self

    def predict(self, X):
        arr = self._to_numpy(X)
        with _dask_joblib():
            raw = self._model.predict(arr)
        return np.where(raw == 1, -1, 1)

    def predict_chunked(self, X):
        """Predict block-by-block on a dask array; delegate to predict otherwise."""
        if not isinstance(X, da.Array) or X.numblocks[0] <= 1:
            return self.predict(X)
        preds = []
        for i in range(X.numblocks[0]):
            block = X.blocks[i].compute()
            if hasattr(block, "toarray"):
                block = block.toarray()
            preds.append(self.predict(np.asarray(block, dtype="float64")))
        return np.concatenate(preds)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

    def decision_function(self, X):
        arr = self._to_numpy(X)
        with _dask_joblib():
            return self._model.decision_function(arr)

class HDBSCAN(BaseDetector):
    """Local reimplementation of PyOD's HDBSCAN detector.
    PyOD removed hdbscan - this is a wrapper to preserve it as an option, using sklearn's implementation.
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
        with _dask_joblib():
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
        with _dask_joblib():
            self.tree_.fit(X)

        return self

    def decision_function(self, X):
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        X = check_array(X)

        with _dask_joblib():
            dist, ind = self.tree_.kneighbors(X)
        weights = 1.0 / (dist + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)

        neighbor_scores = self.decision_scores_[ind]
        scores = np.sum(weights * neighbor_scores, axis=1)
        return scores.ravel()
