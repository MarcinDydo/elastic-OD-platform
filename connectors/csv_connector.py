from dataclasses import dataclass, field
from typing import Optional, Sequence
import dask.dataframe as dd
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