from abc import ABC, abstractmethod
from typing import Optional, Sequence
import dask.dataframe as dd


class DataConnector(ABC):

    name: str
    dataframe: Optional[dd.DataFrame] = None

    @abstractmethod
    def load(self, usecols: Optional[Sequence[str]] = None) -> dd.DataFrame:
        ... #raw df

    def get_df(self, force_reload: bool = False, usecols: Optional[Sequence[str]] = None) -> dd.DataFrame:
        if self.dataframe is None or force_reload:
            self.dataframe = self.load(usecols=usecols) #cache
        return self.dataframe

    def set_df(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        self.dataframe = dataframe
        return self.dataframe
