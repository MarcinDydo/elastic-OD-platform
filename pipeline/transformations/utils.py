import numpy as np
import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix

class WrapStringsTransformer(BaseEstimator, TransformerMixin):
    """Wraps bare string elements into single-element lists.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if hasattr(X, "map"):  # dask Bag
            return X.map(lambda v: [v] if isinstance(v, str) else v)
        return [[v] if isinstance(v, str) else v for v in X]

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if hasattr(X, "toarray"):
            return X.toarray()
        if isinstance(X, da.Array):
            return X.map_blocks(csr_matrix.toarray, dtype="float64")
        if hasattr(X, "todense"):
            return X.todense()
        return X
        

class NanToNumDaskTransformer(BaseEstimator, TransformerMixin):
    """
    Replaces NaN/inf with 0.  Uses da.nan_to_num
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, da.Array):
            result = X.map_blocks(
                lambda b: np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0),
                dtype=X.dtype,
            )
            if any(np.isnan(c) for dim_chunks in result.chunks for c in dim_chunks):
                result = result.persist()
                result.compute_chunk_sizes()
            return result
        arr = np.nan_to_num(np.asarray(X, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
        return da.from_array(arr, chunks=arr.shape)
