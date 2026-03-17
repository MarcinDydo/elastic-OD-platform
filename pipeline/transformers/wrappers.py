import logging
import re
import pandas as pd 
import numpy as np
from pyod.models.ecod import ECOD
from sklearn.cluster import DBSCAN
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from dask_ml.feature_extraction.text import CountVectorizer
from sklearn.ensemble import IsolationForest
import dask.array as da
from dask_ml.decomposition import TruncatedSVD
import dask.dataframe as dd

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

class DBSCANWrapper(BaseEstimator):
    """wrapper around DBSCAN.
    Normalises labels so that -1 = outlier and 1 = inlier, matching
    the convention used by sklearn.
    DBSCAN is transductive — fit() is a no-op, the expensive O(n²) distance computation in predict()
    """

    def __init__(self, eps=0.5, min_samples=5, metric="euclidean", algorithm="auto"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        arr = np.asarray(X, dtype="float64")
        arr = np.nan_to_num(arr, nan=0.0)
        model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm,
        )
        raw_labels = model.fit_predict(arr)
        self.labels_ = np.where(raw_labels == -1, -1, 1)
        return self.labels_

    def fit_predict(self, X, y=None):
        return self.predict(X)

    def decision_function(self, X):
        return np.zeros(np.asarray(X).shape[0])

class ECODWrapper(BaseEstimator):
    """wrapper around pyod ECOD.
    Converts pyod convention (1 = outlier) to sklearn convention
    (-1 = outlier, 1 = inlier).

    The ``contamination`` parameter accepts either a float (pyod default)
    or a pythresh Thresholder instance (e.g. ``FILTER()``).  When a
    thresholder is passed, ECOD is fitted with a placeholder contamination
    and the thresholder's ``eval()`` method determines the final labels.
    """

    def __init__(self, contamination=0.1):
        self.contamination = contamination

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype="float64")
        arr = np.nan_to_num(arr, nan=0.0)

        use_thresholder = hasattr(self.contamination, "eval")

        if use_thresholder:
            self._model = ECOD(contamination=0.5)
            self._model.fit(arr)
            self.decision_scores_ = self._model.decision_scores_
            raw_labels = self.contamination.eval(self.decision_scores_)
            self.labels_ = np.where(np.asarray(raw_labels) == 1, -1, 1)
        else:
            self._model = ECOD(contamination=self.contamination)
            self._model.fit(arr)
            self.labels_ = np.where(self._model.labels_ == 1, -1, 1)
            self.decision_scores_ = self._model.decision_scores_

        return self

    def predict(self, X):
        arr = np.asarray(X, dtype="float64")
        arr = np.nan_to_num(arr, nan=0.0)
        raw = self._model.predict(arr)
        return np.where(raw == 1, -1, 1)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

    def decision_function(self, X):
        arr = np.asarray(X, dtype="float64")
        arr = np.nan_to_num(arr, nan=0.0)
        return self._model.decision_function(arr)

class IsolationForestWrapper(BaseEstimator):
    """wrapper around IsolationForest.
    Stores labels_ and decision_scores_ during fit(), matching the
    pattern used by DBSCANWrapper and ECODWrapper.
    """

    def __init__(self, n_estimators=100, contamination=0.1):
        self.n_estimators = n_estimators
        self.contamination = contamination

    def fit(self, X, y=None):
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
        )
        self.labels_ = self._model.fit_predict(X)
        self.decision_scores_ = self._model.decision_function(X)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

    def decision_function(self, X):
        return self._model.decision_function(X)
    
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


