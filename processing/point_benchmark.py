"""Playbook for benchmarking point anomaly detection — config-driven outlier detection with grid expansion.
"""
from dask_ml.model_selection import GridSearchCV
import dask.array as da
import gc
import json
import logging
import os
import time
import ctypes
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
import pandas as pd
from dask.distributed import Client
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)

from connectors.csv_connector import CSVConnector
from pipeline.builder import TransformerBuilder
from pipeline.transformations.wrappers import (
    AutoencoderWrapper,
    HDBSCAN,
    TfidfVectorizerWrapper,
)

logger = logging.getLogger(__name__)

OVERRIDES: Dict[str, type] = {
    "sklearn.feature_extraction.text.TfidfVectorizer": TfidfVectorizerWrapper,
    "pipeline.transformers.wrappers.AutoencoderWrapper": AutoencoderWrapper,
    "pyod.models.hdbscan.HDBSCAN": HDBSCAN,
}

Y_CONFIG = os.environ.get("Y_NORMAL_LABELS", "class:Normal").split(":")

def _anomaly_recall(y_true, y_pred):
    y_pred_binary = (np.asarray(y_pred) == -1).astype("int8") #ones are where the model predicted anomaly
    return recall_score(y_true, y_pred_binary, zero_division=0)

SCORING = make_scorer(_anomaly_recall)

# Helpers
def _trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

def _chunked_predict(estimator, X_dask):
    """Predict block-by-block when X_dask has multiple partitions."""
    if X_dask.numblocks[0] <= 1:
        arr = X_dask.compute()
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        return estimator.predict(np.asarray(arr, dtype="float64"))

    predictions = []
    for i in range(X_dask.numblocks[0]):
        block = X_dask.blocks[i].compute()
        if hasattr(block, "toarray"):
            block = block.toarray()
        predictions.append(estimator.predict(np.asarray(block, dtype="float64")))

    return np.concatenate(predictions)

def _build_summary(
    task_name: str,
    trained_obj,
    ground_truth: Optional[pd.Series] = None,
) -> Dict[str, Any]:

    is_search = isinstance(trained_obj, GridSearchCV)

    if is_search:
        estimator = trained_obj.best_estimator_.steps[-1][1]
        best_params = trained_obj.best_params_
        best_score = trained_obj.best_score_
    else:
        estimator = trained_obj.steps[-1][1]
        best_params = None
        best_score = None

    labels = getattr(estimator, "labels_", None)
    scores = getattr(estimator, "decision_scores_", None)

    labels_list = list(labels) if labels is not None else []
    n_samples = len(labels_list)
    anomaly_positions = [i for i, l in enumerate(labels_list) if l == -1]

    summary: Dict[str, Any] = {
        "name": task_name,
        "n_samples": n_samples,
        "n_anomalies": len(anomaly_positions),
    }
    if best_params is not None:
        summary["best_params"] = str(best_params)
        summary["best_score"] = best_score

    if is_search:
        cv_res = getattr(trained_obj, "cv_results_", None)
        if cv_res is not None:
            relevant_keys = [
                k for k in cv_res
                if k in ("params", "mean_test_score", "std_test_score",
                         "rank_test_score", "mean_fit_time", "std_fit_time")
                or k.startswith("param_")
            ]
            filtered = {}
            for k in relevant_keys:
                val = cv_res[k]
                if hasattr(val, "tolist"):
                    val = val.tolist()
                filtered[k] = val
            summary["cv_results"] = json.dumps(filtered)

    # Ground-truth metrics
    if ground_truth is not None and labels_list:
        y_true = ((ground_truth != "Normal").astype("int8")).to_dask_array(lengths=True).compute()
        y_pred = (pd.Series(labels_list) == -1).astype("int8").to_numpy()

        if y_true.shape[0] == y_pred.shape[0]:
            summary["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
            summary["precision"] = round(precision_score(y_true, y_pred, zero_division=0), 4)
            summary["recall"] = round(recall_score(y_true, y_pred, zero_division=0), 4)
            summary["f1_score"] = round(f1_score(y_true, y_pred, zero_division=0), 4)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            summary["confusion_matrix"] = {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }

    if scores is not None:
        scores_list = list(scores)
        if scores_list:
            summary["score_mean"] = float(np.mean(scores_list))
            summary["score_min"] = float(np.min(scores_list))
            summary["score_max"] = float(np.max(scores_list))

    return summary

# Build experiment
def build_dags() -> Tuple[Any, Dict[str, Dict], Dict[str, Any]]:
    """Load data and build all strategy pipelines (nothing computed yet)."""
    builder = TransformerBuilder(overrides=OVERRIDES)

    Y_config = os.environ.get("Y_NORMAL_LABELS", "class:Normal").split(":")
    csv_columns = builder.required_columns(Y_config[0])

    connector = CSVConnector()
    ddf = connector.load(usecols=csv_columns)
    jobs = builder.build_all(scoring_fn=SCORING)

    total_tasks = sum(
        len(j["feature_pipes"]) * (len(j["pipelines"]) + len(j["searches"]))
        for j in jobs.values()
    )
    build_info = {
        "csv_path": connector.path,
        "config_path": os.getenv("CONFIG_PATH"),
        "required_columns": csv_columns,
        "strategies": list(jobs.keys()),
        "total_tasks": total_tasks,
        "y_config": Y_config,
    }
    return ddf, jobs, build_info

# Run experiment
def run(
    client: Client,
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Execute playbook 2: sequential per-strategy computation.
    Each feature pipe the feature array is computed once
    Many estimator objects (pipelines/searches) are trained on the same feature array.
    """
    ddf, jobs, build_info = build_dags()
    logger.info("Playbook 2 build info:\n%s", json.dumps(build_info, indent=2, default=str))

    #persist data on workers
    ddf = client.persist(ddf)

    # Materialize once on client for anomaly-row slicing per search.
    full_df = ddf.compute()

    ground_truth = ddf[build_info["y_config"][0]]
    Y_dask = (ground_truth != build_info["y_config"][1]).astype("int8").to_dask_array(lengths=True) #ones are everything that is not labeled Normal
    Y_dask = client.persist(Y_dask)

    # Incremental writers — start each run fresh
    result_connector = CSVConnector()
    results_path = os.environ.get("RESULTS_PATH")
    if results_path and os.path.exists(results_path):
        os.remove(results_path)
    target_path = os.environ.get("TARGET_PATH")

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    #execute ml jobs to find outliers 
    for strategy_name, job in jobs.items():
        feature_pipes = job["feature_pipes"]
        pipelines = job["pipelines"]
        searches = job["searches"]

        n_tasks = len(feature_pipes) * (len(pipelines) + len(searches))
        logger.info(
            "=== Strategy: %s  (%d feature pipes x (%d pipes + %d searches) = %d tasks) ===",
            strategy_name, len(feature_pipes), len(pipelines), len(searches), n_tasks,
        )

        if not feature_pipes:
            logger.warning("Strategy '%s' has no feature pipes, skipping.", strategy_name)
            continue

        strategy_results = []

        #first compute features
        for feat_desc, col_pipes in feature_pipes:
            # 1) Fit-transform ALL column pipelines, collect arrays
            logger.debug("  Fitting composite feature pipe: %s", feat_desc)
            t0 = time.perf_counter()
            col_arrays = []
            for col_name, pipe in col_pipes.items():
                feature_arr = pipe.fit_transform(ddf[col_name].to_bag())
                if isinstance(feature_arr, da.Array):
                    feature_arr = feature_arr.compute()
                elif hasattr(feature_arr, "compute"):
                    feature_arr = feature_arr.compute()
                if hasattr(feature_arr, "toarray"):
                    feature_arr = feature_arr.toarray()
                arr = np.asarray(feature_arr, dtype="float64")
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                col_arrays.append(arr)

            # Concatenate all columns horizontally
            X_combined = np.hstack(col_arrays)
            X_dask = client.persist(da.from_array(X_combined, chunks="64MB"))
            del col_arrays, X_combined
            elapsed = time.perf_counter() - t0
            logger.info(
                "  Composite feature '%s' computed in %.1fs  shape=%s",
                feat_desc, elapsed, getattr(X_dask, "shape", "?"),
            )

            # 2) Plain pipelines → batch via dask.delayed TODO
            if pipelines:
                pass

            # 3) Searches → persist feature array as numpy array on workers but let the GridSearchCV handle that
            for s_desc, search in searches:
                if not isinstance(search, GridSearchCV):
                    logger.warning("Grid Search CV is not dask and may cause OOM errors. "
                    )
                task_name = f"{strategy_name}__{feat_desc}__{s_desc}"
                logger.debug("    Running GridSearchCV on client: %s", task_name)
                t0 = time.perf_counter()
                search.fit(X_dask, y=Y_dask) #Find best (seach is dask compatible)
                best_est = search.best_estimator_.steps[-1][1]
                best_est.labels_ = _chunked_predict(best_est, X_dask)
                client.run(gc.collect)
                elapsed = time.perf_counter() - t0
                logger.info("    Completed in %.1fs: %s", elapsed, task_name)

                # Build summary, append to RESULTS_PATH
                summary = _build_summary(task_name, search, ground_truth)
                summary["strategy"] = strategy_name
                strategy_results.append(summary)
                result_connector.save_row(summary)
                logger.info(
                    "  %s  n_anomalies=%d  accuracy=%.4f  precision=%.4f  recall=%.4f  f1=%.4f",
                    task_name,
                    summary.get("n_anomalies", 0),
                    summary.get("accuracy", 0.0),
                    summary.get("precision", 0.0),
                    summary.get("recall", 0.0),
                    summary.get("f1_score", 0.0),
                )

                # Overwrite TARGET_PATH with this grid's anomaly rows (full columns)
                if target_path:
                    anomaly_rows = full_df[best_est.labels_ == -1]
                    anomaly_rows.to_csv(target_path, index=False, sep=";", mode="w")
                    logger.info("    Wrote %d anomaly rows to %s", len(anomaly_rows), target_path)

            if searches:
                del X_dask
                client.run(gc.collect)
                client.run(_trim_memory)

            gc.collect()
            client.run(gc.collect)

        logger.debug("Strategy '%s' complete.\n", strategy_name)
        all_results[strategy_name] = strategy_results

    del Y_dask
    return all_results
