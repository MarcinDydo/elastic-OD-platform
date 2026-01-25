from typing import Any, Dict
import psutil
import dask.dataframe as dd
from dask.delayed import delayed
from processing.reduce_interface import ReduceInterface
from sklearn.cluster import DBSCAN
import psutil
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd


def _run_dbscan(frame: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
    arr = frame.to_numpy(dtype='float64')
    arr = np.nan_to_num(arr, nan=0.0)
    
    model = DBSCAN(
        eps=params.get('eps', 0.5),
        min_samples=params.get('min_samples', 5)
    )
    labels = model.fit_predict(arr)
    
    return labels


class TimeSeriesOutlierReducer(ReduceInterface):
    def __init__(self, name: str, params: Dict[str, Any]):
        self.result = None
        self.features = []
        self.params = params
        self.name = name

    def check_params(self) -> bool:
        return True

    def reduce(self, lddf: Dict[str, dd.DataFrame]):
        if not self.check_params():
            raise ValueError(f"Invalid parameters for TimeSeriesOutlierReducer '{self.params}'")
        
        ddf = self.prepare(lddf)  # dd.DataFrame from concat
        self.result = delayed(_run_dbscan)(ddf, self.params)
        return self.result