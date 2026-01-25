from typing import Any, Dict, List

import dask.dataframe as dd
from dask.delayed import delayed
from dask.distributed import  futures_of
from processing.reduce_interface import ReduceInterface
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd


def _run_iforest(frame: dd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    model = IsolationForest(**params)
    model.fit(frame)
    scores = model.decision_function(frame)
    labels = model.predict(frame)
    return labels, scores


class IsolationForestReducer(ReduceInterface):
    def __init__(self, name: str, params: Dict[str, Any]):
        self.result = None
        self.features = []
        self.params = params
        self.name = name
   
    def check_params(self) -> bool: #TODO:
        # if self.params.
        return True

    def reduce(self, lddf: Dict[str, dd.DataFrame]):
        if not self.check_params():
            raise ValueError(f"Invalid parameters for IsolationForestReducer '{self.params}'")
        ddf = self.prepare(lddf)
        self.result = delayed(_run_iforest)(ddf, self.params)
        return self.result
