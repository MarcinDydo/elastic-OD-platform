import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FillNaTransformer(BaseEstimator, TransformerMixin):
    """Replaces NaN/None with a fill value and casts to str.

    Works with both pandas Series and dask Series.
    """
    _dask_native = True

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


class ToDenseTransformer(BaseEstimator, TransformerMixin):
    """Converts sparse matrices to dense numpy arrays.

    Designed as an sklearn Pipeline step placed before estimators that
    require dense input (DBSCAN, ECOD, etc.).  Kept as a separate class
    so that dimensionality-reduction steps (e.g. TruncatedSVD) can be
    composed or substituted later.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.asarray(X)

