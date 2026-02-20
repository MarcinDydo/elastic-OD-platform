from typing import Any, Dict

import dask.dataframe as dd
from dask.delayed import delayed
from processing.reduce_interface import ReduceInterface
from pyod.models.ecod import ECOD
import numpy as np
import pandas as pd


def _run_ecod(frame: pd.DataFrame, params: Dict[str, Any]):
    arr = frame.to_numpy(dtype='float64')
    arr = np.nan_to_num(arr, nan=0.0)

    model = ECOD(**params)
    model.fit(arr)
    scores = model.decision_scores_
    labels = model.labels_  # 0 = inlier, 1 = outlier

    # Convert pyod convention (1=outlier) to sklearn convention (-1=outlier)
    labels = np.where(labels == 1, -1, 1)
    return labels, scores


class ECODReducer(ReduceInterface):
    def __init__(self, name: str, params: Dict[str, Any]):
        self.result = None
        self.features = []
        self.params = params
        self.name = name

    def check_params(self) -> bool:
        return True

    def reduce(self, lddf: Dict[str, dd.DataFrame]):
        if not self.check_params():
            raise ValueError(f"Invalid parameters for ECODReducer '{self.params}'")
        ddf = self.prepare(lddf)
        self.result = delayed(_run_ecod)(ddf, self.params)
        return self.result
