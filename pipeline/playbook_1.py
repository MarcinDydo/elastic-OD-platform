import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

import dask
from dask.distributed import Client

from connectors.csv_connector import CSVConnector
from processing.mappers.transformer import TransformerMapper
from processing.reduce_interface import ReduceInterface
from processing.map_interface import MapInterface
from processing.reducers.isolation_forest import IsolationForestReducer
from processing.reducers.outlier import TimeSeriesOutlierReducer


def _csv_columns() -> List[str]:
    path = os.getenv("REDUCE_CONFIG_PATH")
    config = json.load(open(path, "r", encoding="utf-8"))
    required_columns = set()
    for reducer_conf in config.values():
        features = reducer_conf.get("features", {})
        required_columns.update(features.keys())
    if not path:
        raise RuntimeError("REDUCE_CONFIG_PATH is required")
    required_columns.update(["class"])
    return list(required_columns) #unique specific columns 

def _print_json(title: str, payload: object) -> None:
    def _coerce(value: object) -> object:
        if isinstance(value, dict):
            return {key: _coerce(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [_coerce(val) for val in value]
        if hasattr(value, "__dict__"):
            return {key: _coerce(val) for key, val in vars(value).items()}
        return value
    print(title)
    print(json.dumps(_coerce(payload), indent=2, default=str))

def _build_reducers(
    config_path: Optional[str] = None,
) -> Tuple[List[ReduceInterface], Dict[str, object]]:
    config = ReduceInterface.load_config(config_path)
    reducers: List[ReduceInterface] = []
    encodings = set()
    for name, conf in config.items():
        features = conf.get("features", {}) #dict of column name to encoding strategy
        encodings.update([(f,s) for f, s in features.items()])
        params = dict(conf.get("params", {}))
        match conf.get("algo"):
            case "dbscan":
                reducer = TimeSeriesOutlierReducer(name=name, params=params)
            case "isolation_forest":
                reducer = IsolationForestReducer(name=name, params=params)
            case _:
                raise ValueError(f"Unsupported reducer algo: {conf.get('algo')}")
        reducer.features = [f"{f}_{s}" for f, s in features.items()]
        reducers.append(reducer)
    return reducers, list(encodings)  #two lists 

def _build_summary(reducer: ReduceInterface, result: Any, ddf: Any) -> Dict[str, object]:
    if isinstance(reducer, IsolationForestReducer):
        algo = "isolation_forest"
        labels = result[0]
        scores = result[1]
    elif isinstance(reducer, TimeSeriesOutlierReducer):
        algo = "dbscan"
        scores = None
        labels = result
    else:
        algo = type(reducer).__name__
        scores = None
        labels = result

    labels_list = list(labels) if labels is not None else []
    n_samples = len(labels_list)
    anomaly_positions = [idx for idx, label in enumerate(labels_list) if label == -1]

    summary: Dict[str, object] = {
        "name": reducer.name,
        "algo": type(reducer).__name__,
        "params": getattr(reducer, "params", None),
        "n_samples": n_samples,
        "n_anomalies": len(anomaly_positions),
        "anomaly_indices": str(anomaly_positions[:100]),
    }

    # Calculate accuracy metrics if ground truth is available
    if ddf is not None and 'class' in ddf.columns:
        ground_truth = ddf['class'].compute()
        
        # Convert to binary: 1 for anomaly, 0 for normal
        y_true = (ground_truth == "Anomalous").astype(int).values
        y_pred = (labels_list == -1).astype(int) if isinstance(labels_list, pd.Series) else [1 if l == -1 else 0 for l in labels_list]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        summary["accuracy"] = round(accuracy, 4)
        summary["precision"] = round(precision, 4)
        summary["recall"] = round(recall, 4)
        summary["f1_score"] = round(f1, 4)
        summary["confusion_matrix"] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }

    if scores is not None:
        scores_list = list(scores)
        if scores_list:
            summary["score_mean"] = float(sum(scores_list) / len(scores_list))
            summary["score_min"] = float(min(scores_list))
            summary["score_max"] = float(max(scores_list))
        if scores_list and anomaly_positions and len(scores_list) == n_samples:
            min_pos = min(anomaly_positions, key=lambda pos: scores_list[pos])
            max_pos = max(anomaly_positions, key=lambda pos: scores_list[pos])
            summary["anomaly_score_min"] = float(scores_list[min_pos])
            summary["anomaly_score_max"] = float(scores_list[max_pos])
        
    return summary

def build_graph() -> Tuple[object, List[ReduceInterface], Dict[str, object]]:
    try:
        mapper_config_path = os.environ["MAPPER_CONFIG_PATH"]
        reducer_config_path = os.environ["REDUCE_CONFIG_PATH"]
        csv_blocksize = os.environ["CSV_BLOCKSIZE"]
        csv_assume_missing = bool(os.environ["CSV_ASSUME_MISSING"])
    except KeyError as exc:
        raise RuntimeError(f"{exc.args[0]} is required in environment") from exc

    csv_columns = _csv_columns()
    reducers, encodings = _build_reducers(reducer_config_path) #this is list of reducer objects each object has a prepared callable reduce function
    mapper = TransformerMapper(mapper_config_path, encodings) #this is one mapper object it has a prepared callable map function
    
    connector = CSVConnector(
        path= os.getenv("CSV_PATH"),
        blocksize=csv_blocksize,
        assume_missing=csv_assume_missing,
    )# this is one connector object with load function that will
    # _print_json("1",reducers)
    # print(encodings)
    # _print_json("2",mapper)
    # print(connector)   
    ddf = connector.load(usecols=csv_columns) #delayed dataframe object

    ddfs = mapper.map(ddf) #map columns from delayed dataframe to a dict of encoded delayed dataframes  

    # if ddfs is not None:
    #     print(ddfs)
    #     dask.visualize(*ddfs.values(), filename=os.getenv("GRAPH_PATH"))
    #     return

    for reducer in reducers:
        reducer.reduce(ddfs)

    build_info = {
        "csv_path": connector.path,
        "mapper_config_path": mapper_config_path,
        "reducer_config_path": reducer_config_path,
        "required_columns": csv_columns,
        "reducers": [reducer.name for reducer in reducers],
        "csv_blocksize": csv_blocksize,
        "csv_assume_missing": csv_assume_missing,
    }
    
    return ddf, reducers, build_info #Since we are returning a reference to reducers we should also return reference to their results inside, which will store future object but not
    #returning otriginal ddf to see the real value of outliers

def run(
    client: Client,
    show_run_parameters: bool = True,
    show_reduce_results: bool = True,
) -> Optional[Dict[str, object]]:

    ddf, reducers, build_info = build_graph()  #a dask dataframe and list of reducer objects

    delayed_results = [reducer.result for reducer in reducers]
    #print(delayed_results)
    if show_run_parameters:
        _print_json("playbook_1 build info:", build_info)
        dask.visualize(*delayed_results, filename=os.getenv("GRAPH_PATH"))
    
    futures = None
    if not show_reduce_results or len(delayed_results) ==0:
        return None
    else:
        futures  = client.compute(delayed_results)
    
    results = client.gather(futures)
    output: Dict[str, List[Dict[str, object]]] = {}
    for reducer, result in zip(reducers, results):
        summary: Dict[str, object] = _build_summary(reducer, result, ddf)
        output[reducer.name] = [summary]

    _print_json("reduce results:", output)
