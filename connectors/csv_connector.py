from dataclasses import dataclass, field
from typing import Optional, Sequence, Union
import os
import dask.dataframe as dd
import pandas as pd
from dask.delayed import delayed
from .connector_interface import DataConnector


@dataclass
class CSVConnector(DataConnector):
    path: Optional[str] = None
    blocksize: Optional[str] = None
    assume_missing: Optional[bool] = None
    usecols: Optional[Sequence[str]] = None
    dataframe: Optional[dd.DataFrame] = None

    def __post_init__(self):
        if self.path is None:
            self.path = os.getenv("CSV_PATH")
        if self.blocksize is None:
            self.blocksize = os.getenv("CSV_BLOCKSIZE", "64MB")
        if self.assume_missing is None:
            env_val = os.getenv("CSV_ASSUME_MISSING", "")
            self.assume_missing = env_val.lower() in ("true", "1", "yes") if env_val else False

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
        """Save a DataFrame to CSV. Uses *path* or RESULTS_PATH env var."""
        out = path or os.environ.get("RESULTS_PATH")
        if not out:
            raise RuntimeError("RESULTS_PATH is required for saving results.")
        if isinstance(df, dd.DataFrame):
            df = df.compute()
        df.to_csv(out, index=False, sep=";")
        return out

    def save_row(self, row: Union[dict, pd.Series], path: Optional[str] = None) -> str:
        """Append a single row to RESULTS_PATH (writes header on first call)."""
        out = path or os.environ.get("RESULTS_PATH")
        if not out:
            raise RuntimeError("RESULTS_PATH is required for saving metrics rows.")
        if isinstance(row, dict):
            df = pd.json_normalize(row, sep="_")
        else:
            df = pd.DataFrame([row])
        header = not os.path.exists(out)
        df.to_csv(out, index=False, sep=";", mode="a", header=header)
        return out