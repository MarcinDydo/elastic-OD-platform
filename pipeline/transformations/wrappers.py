import logging
import math
import os
import re
from collections import Counter
from contextlib import contextmanager
import pandas as pd
import numpy as np
from joblib import parallel_backend
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher as _SklearnFeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
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
    return os.getenv("DASK_DATAFRAME__BACKEND") == "cudf"


class TfidfVectorizerWrapper(BaseEstimator, TransformerMixin):
    """TF-IDF transformer with dual CPU/GPU backend.

    GPU path (DASK_DATAFRAME__BACKEND=cudf): dask_ml CountVectorizer produces
    a sparse dask matrix, then cuml.dask TfidfTransformer weighs it across
    workers.  CPU path delegates to sklearn's TfidfVectorizer under the dask
    joblib backend so tokenization parallelises on the cluster.
    """

    def __init__(
        self,
        token_pattern=None,
        lowercase=True,
        max_features=None,
        analyzer="word",
        ngram_range=(1, 1),
        min_df=1,
        max_df=1.0,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ):
        self.token_pattern = token_pattern
        self.lowercase = lowercase
        self.max_features = max_features
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def _count_kwargs(self):
        kwargs = {
            "lowercase": self.lowercase,
            "analyzer": self.analyzer,
            "ngram_range": tuple(self.ngram_range),
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_features": self.max_features,
        }
        if self.token_pattern is not None:
            kwargs["token_pattern"] = self.token_pattern
        return kwargs

    @staticmethod
    def _ensure_concrete(X):
        if hasattr(X, "compute"):
            return X.compute()
        return X

    def _fit_gpu(self, X):
        from dask_ml.feature_extraction.text import CountVectorizer as DaskCountVectorizer
        from cuml.dask.feature_extraction.text import TfidfTransformer as CumlTfidfTransformer

        self._count_ = DaskCountVectorizer(**self._count_kwargs())
        counts = self._count_.fit_transform(X)
        self._tfidf_ = CumlTfidfTransformer()
        return self._tfidf_.fit_transform(counts)

    def _transform_gpu(self, X):
        counts = self._count_.transform(X)
        return self._tfidf_.transform(counts)

    def _fit_cpu(self, X):
        kwargs = self._count_kwargs()
        kwargs.update({
            "norm": self.norm,
            "use_idf": self.use_idf,
            "smooth_idf": self.smooth_idf,
            "sublinear_tf": self.sublinear_tf,
        })
        self._sk_ = TfidfVectorizer(**kwargs)
        with _dask_joblib():
            return self._sk_.fit_transform(self._ensure_concrete(X))

    def _transform_cpu(self, X):
        with _dask_joblib():
            return self._sk_.transform(self._ensure_concrete(X))

    def fit(self, X, y=None):
        if _gpu_backend():
            self._fit_gpu(X)
        else:
            self._fit_cpu(X)
        return self

    def fit_transform(self, X, y=None):
        return self._fit_gpu(X) if _gpu_backend() else self._fit_cpu(X)

    def transform(self, X):
        return self._transform_gpu(X) if _gpu_backend() else self._transform_cpu(X)


    
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

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if hasattr(X, "compute"):
            X = X.compute()

        illegal = self._ILLEGAL_CHARS
        encoded_pat = self._ENCODED_RE

        rows = []
        for text in pd.Series(X).fillna("").astype(str):
            s = text.lower() if self.lowercase else text
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
                alnum / n,
                special / n,
                illegal_count / n,
                encoded / n,
                entropy,
            ])

        return np.array(rows, dtype="float64")


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

    def transform(self, X, y=None):
        check_is_fitted(self, ["ft_model_"])
        if hasattr(X, "compute"):
            X = X.compute()

        wv = self.ft_model_.wv
        dim = wv.vector_size
        pat = self._pat
        rows = []
        for text in pd.Series(X).fillna("").astype(str):
            tokens = pat.findall(text)
            if not tokens:
                rows.append(np.zeros(dim, dtype="float64"))
                continue
            vecs = np.array([wv[t] for t in tokens])
            rows.append(vecs.mean(axis=0))

        return np.array(rows, dtype="float64")


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
