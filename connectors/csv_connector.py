from dataclasses import dataclass, field
from typing import Optional, Sequence, Union
import os
import dask.dataframe as dd
import pandas as pd
from dask.delayed import delayed
from .connector_interface import DataConnector


@dataclass
class CSVConnector(DataConnector):
    path: str
    blocksize: Optional[str] = "64MB"
    assume_missing: bool = False
    usecols: Optional[Sequence[str]] = None
    dataframe: Optional[dd.DataFrame] = None

    def load(self, usecols: Optional[Sequence[str]] = None) -> dd.DataFrame:
        if usecols is not None:
            self.usecols = list(usecols)
        self.dataframe = dd.read_csv(
            self.path,
            blocksize=self.blocksize,
            assume_missing=self.assume_missing,
            usecols=self.usecols,
        )
        return self.dataframe #delayed dask.DataFrame object

    def save(self, df: Union[pd.DataFrame, dd.DataFrame], path: Optional[str] = None) -> str:
        """Save a DataFrame to CSV. Uses *path* or CSV_RESULT_PATH env var."""
        out = path or os.environ.get("CSV_RESULT_PATH")
        if not out:
            raise RuntimeError("CSV_RESULT_PATH is required for saving results.")
        if isinstance(df, dd.DataFrame):
            df = df.compute()
        df.to_csv(out, index=False)
        return out