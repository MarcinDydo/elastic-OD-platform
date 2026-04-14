import numpy as np
import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin


class FillNaTransformer(BaseEstimator, TransformerMixin):
    """Replaces NaN/None with a fill value and casts to str.

    Works with both pandas Series and dask Series.
    """

    def __init__(self, fill_value=""):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if hasattr(X, "fillna"):  # pandas / dask dataframe/series
            return X.fillna(self.fill_value).astype(str)

        # dask bag
        if hasattr(X, "map"):
            return X.map(
                lambda v: str(self.fill_value if v is None else v)
            )

        # fallback
        return str(self.fill_value if X is None else X)


class WrapStringsTransformer(BaseEstimator, TransformerMixin):
    """Wraps bare string elements into single-element lists.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if hasattr(X, "map"):  # dask Bag
            return X.map(lambda v: [v] if isinstance(v, str) else v)
        return [[v] if isinstance(v, str) else v for v in X]


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
