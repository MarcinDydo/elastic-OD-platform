"""Sequence anomaly detection playbook.
Aggregates consecutive feature vectors into fixed-size windows (mean or sum),
"""
from dask_ml.model_selection import GridSearchCV
import dask.array as da
import dask.dataframe as dd
import gc
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dask.distributed import Client

from connectors.csv_connector import CSVConnector
from processing.point_benchmark import (
    _build_summary,
    _chunked_predict,
    _trim_memory,
    build_dags,
)

logger = logging.getLogger(__name__)

SEQUENCE_CONFIG = {
    "size": 80,
    "method": "mean",
    "overlap": 0,
}


def _window_slices(n_rows: int, size: int, overlap: int) -> List[Tuple[int, int]]:
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap must be strictly less than size")
    if n_rows < size:
        return []
    return [(s, s + size) for s in range(0, n_rows - size + 1, step)]


def _aggregate_features(
    X: np.ndarray,
    size: int,
    overlap: int,
    method: str,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    slices = _window_slices(X.shape[0], size, overlap)
    if not slices:
        return np.empty((0, X.shape[1]), dtype=X.dtype), slices
    reducer = np.sum if method == "sum" else np.mean
    rows = [reducer(X[s:e], axis=0) for s, e in slices]
    return np.vstack(rows).astype("float64"), slices


def _aggregate_labels(
    y: np.ndarray,
    slices: List[Tuple[int, int]],
) -> np.ndarray:
    if not slices:
        return np.empty((0,), dtype="int8")
    return np.array([int(y[s:e].max()) for s, e in slices], dtype="int8")


def _expand_labels_to_rows(
    seq_labels: np.ndarray,
    slices: List[Tuple[int, int]],
    n_rows: int,
) -> np.ndarray:
    row_flags = np.zeros(n_rows, dtype="int8")
    for (s, e), lab in zip(slices, seq_labels):
        if lab == -1:
            row_flags[s:e] = -1
    return row_flags


def _ground_truth_series(y_seq: np.ndarray, normal_label: str) -> dd.Series:
    """Wrap sequence labels into the same shape _build_summary expects."""
    series = pd.Series(np.where(y_seq == 1, "Anomaly", normal_label))
    return dd.from_pandas(series, npartitions=1)


def run(client: Client) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    ddf, jobs, build_info = build_dags()
    build_info["sequence_config"] = SEQUENCE_CONFIG
    logger.info("Sequence playbook build info:\n%s", json.dumps(build_info, indent=2, default=str))

    ddf = client.persist(ddf)
    full_df = ddf.compute()

    y_col, normal_label = build_info["y_config"][0], build_info["y_config"][1]
    ground_truth = ddf[y_col]
    y_np = (ground_truth != normal_label).astype("int8").to_dask_array(lengths=True).compute()

    result_connector = CSVConnector()
    results_path = os.environ.get("RESULTS_PATH")
    if results_path and os.path.exists(results_path):
        os.remove(results_path)
    target_path = os.environ.get("TARGET_PATH")

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for strategy_name, job in jobs.items():
        feature_pipes = job["feature_pipes"]
        pipelines = job["pipelines"]
        searches = job["searches"]

        n_tasks = len(feature_pipes) * (len(pipelines) + len(searches))
        logger.info(
            "=== Sequence strategy: %s  (%d feature pipes x (%d pipes + %d searches) = %d tasks) ===",
            strategy_name, len(feature_pipes), len(pipelines), len(searches), n_tasks,
        )

        if not feature_pipes:
            logger.warning("Strategy '%s' has no feature pipes, skipping.", strategy_name)
            continue

        strategy_results = []

        for feat_desc, col_pipes in feature_pipes:
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

            X_combined = np.hstack(col_arrays)
            del col_arrays

            X_seq, slices = _aggregate_features(
                X_combined,
                size=SEQUENCE_CONFIG["size"],
                overlap=SEQUENCE_CONFIG["overlap"],
                method=SEQUENCE_CONFIG["method"],
            )
            Y_seq = _aggregate_labels(y_np, slices)
            elapsed = time.perf_counter() - t0
            logger.info(
                "  Composite feature '%s' aggregated in %.1fs  rows %d -> sequences %d  shape=%s",
                feat_desc, elapsed, X_combined.shape[0], X_seq.shape[0], X_seq.shape,
            )
            del X_combined

            if X_seq.shape[0] == 0:
                logger.warning(
                    "  No sequences produced (n_rows<%d). Skipping feature pipe.",
                    SEQUENCE_CONFIG["size"],
                )
                continue

            X_dask = client.persist(da.from_array(X_seq, chunks="64MB"))
            Y_dask = client.persist(da.from_array(Y_seq, chunks="auto"))
            seq_ground_truth = _ground_truth_series(Y_seq, normal_label)

            if pipelines:
                pass  # TODO (parity with point_benchmark)

            for s_desc, search in searches:
                if not isinstance(search, GridSearchCV):
                    logger.warning("Grid Search CV is not dask and may cause OOM errors.")
                task_name = f"{strategy_name}__{feat_desc}__{s_desc}"
                logger.debug("    Running GridSearchCV on client: %s", task_name)
                t0 = time.perf_counter()
                search.fit(X_dask, y=Y_dask)
                best_est = search.best_estimator_.steps[-1][1]
                best_est.labels_ = _chunked_predict(best_est, X_dask)
                client.run(gc.collect)
                elapsed = time.perf_counter() - t0
                logger.info("    Completed in %.1fs: %s", elapsed, task_name)

                row_flags = _expand_labels_to_rows(
                    best_est.labels_, slices, len(full_df),
                )
                n_row_anomalies = int((row_flags == -1).sum())

                summary = _build_summary(task_name, search, seq_ground_truth)
                summary["strategy"] = strategy_name
                summary["sequence_config"] = json.dumps(SEQUENCE_CONFIG)
                summary["n_row_anomalies"] = n_row_anomalies
                strategy_results.append(summary)
                result_connector.save_row(summary)
                logger.info(
                    "  %s  n_seq_anomalies=%d  n_row_anomalies=%d  accuracy=%.4f  precision=%.4f  recall=%.4f  f1=%.4f",
                    task_name,
                    summary.get("n_anomalies", 0),
                    n_row_anomalies,
                    summary.get("accuracy", 0.0),
                    summary.get("precision", 0.0),
                    summary.get("recall", 0.0),
                    summary.get("f1_score", 0.0),
                )

                if target_path:
                    anomaly_rows = full_df[row_flags == -1]
                    anomaly_rows.to_csv(target_path, index=False, sep=";", mode="w")
                    logger.info(
                        "    Wrote %d anomaly rows (%d flagged sequences) to %s",
                        len(anomaly_rows),
                        int((best_est.labels_ == -1).sum()),
                        target_path,
                    )

            if searches:
                del X_dask, Y_dask
                client.run(gc.collect)
                client.run(_trim_memory)

            gc.collect()
            client.run(gc.collect)

        logger.debug("Strategy '%s' complete.\n", strategy_name)
        all_results[strategy_name] = strategy_results

    return all_results
