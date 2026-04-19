"""Abstract Playbook interface shared by the four outlier-detection entry points.
contains a bunch of helpers for creating pipelines, summarizing results, and writing output. The concrete"""
from __future__ import annotations

import ctypes
import gc
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client
from dask_ml.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)

from connectors.connector_interface import DataConnector
from connectors.csv_connector import CSVConnector
from pipeline.builder import TransformerBuilder
from pipeline.transformations.wrappers import (
    AutoencoderWrapper,
    HDBSCAN,
    TfidfTransformerWrapper,
)

logger = logging.getLogger(__name__)

OVERRIDES: Dict[str, type] = {
    "sklearn.feature_extraction.text.TfidfVectorizer": TfidfTransformerWrapper,
    "pipeline.transformers.wrappers.AutoencoderWrapper": AutoencoderWrapper,
}


def _anomaly_recall(y_true, y_pred):
    y_pred_binary = (np.asarray(y_pred) == -1).astype("int8")
    return recall_score(y_true, y_pred_binary, zero_division=0)


SCORING = make_scorer(_anomaly_recall)


def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


@dataclass
class FeatureContext:
    """State carried between _prepare_X and _expand_row_mask / finalizers."""
    feat_desc: str
    n_rows: int
    slices: Optional[List[Tuple[int, int]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def task_summary(
    task_name: str,
    strategy: str,
    labels: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    labels = np.asarray(labels)
    summary: Dict[str, Any] = {
        "name": task_name,
        "strategy": strategy,
        "n_samples": int(labels.shape[0]),
        "n_anomalies": int((labels == -1).sum()),
    }
    if extra:
        summary.update(extra)
    return summary


def build_summary(
    task_name: str,
    strategy: str,
    trained_obj,
    labels: np.ndarray,
    ground_truth: Optional[dd.Series] = None,
    normal_label: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    labels = np.asarray(labels)
    labels_list = labels.tolist()
    n_samples = len(labels_list)
    anomaly_positions = [i for i, lab in enumerate(labels_list) if lab == -1]

    summary: Dict[str, Any] = {
        "name": task_name,
        "strategy": strategy,
        "n_samples": n_samples,
        "n_anomalies": len(anomaly_positions),
    }

    if isinstance(trained_obj, GridSearchCV):
        estimator = trained_obj.best_estimator_.steps[-1][1]
        summary["best_params"] = str(trained_obj.best_params_)
        summary["best_score"] = trained_obj.best_score_
        cv_res = getattr(trained_obj, "cv_results_", None)
        if cv_res is not None:
            relevant = [
                k for k in cv_res
                if k in ("params", "mean_test_score", "std_test_score",
                         "rank_test_score", "mean_fit_time", "std_fit_time")
                or k.startswith("param_")
            ]
            filtered: Dict[str, Any] = {}
            for k in relevant:
                val = cv_res[k]
                if hasattr(val, "tolist"):
                    val = val.tolist()
                filtered[k] = val
            summary["cv_results"] = json.dumps(filtered)
    elif trained_obj is not None and hasattr(trained_obj, "steps"):
        estimator = trained_obj.steps[-1][1]
    else:
        estimator = None

    scores = getattr(estimator, "decision_scores_", None) if estimator is not None else None

    if ground_truth is not None and labels_list:
        compare_value = normal_label if normal_label is not None else "Normal"
        y_true = ((ground_truth != compare_value).astype("int8")
                  .to_dask_array(lengths=True).compute())
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

    if extra:
        summary.update(extra)
    return summary


def vote_labels(labels_list: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Majority vote across models. Returns -1/1 array where at least
    `threshold` of models flagged -1.
    """
    stack = np.stack(labels_list)
    anomaly_count = (stack == -1).sum(axis=0)
    n = len(labels_list)
    return np.where(anomaly_count >= threshold * n, -1, 1).astype("int8")



WINDOW_DEFAULTS = {
    "size": 80,
    "method": "mean",
    "overlap": 0,
}


def window_slices(n_rows: int, size: int, overlap: int) -> List[Tuple[int, int]]:
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap must be strictly less than size")
    if n_rows < size:
        return []
    return [(s, s + size) for s in range(0, n_rows - size + 1, step)]


def aggregate_features(
    X: np.ndarray,
    size: int,
    overlap: int,
    method: str,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    slices = window_slices(X.shape[0], size, overlap)
    if not slices:
        return np.empty((0, X.shape[1]), dtype=X.dtype), slices
    reducer = np.sum if method == "sum" else np.mean
    rows = [reducer(X[s:e], axis=0) for s, e in slices]
    return np.vstack(rows).astype("float32"), slices


def aggregate_labels(
    y: np.ndarray,
    slices: List[Tuple[int, int]],
) -> np.ndarray:
    if not slices:
        return np.empty((0,), dtype="int8")
    return np.array([int(y[s:e].max()) for s, e in slices], dtype="int8")


def expand_labels_to_rows(
    seq_labels: np.ndarray,
    slices: List[Tuple[int, int]],
    n_rows: int,
) -> np.ndarray:
    row_flags = np.zeros(n_rows, dtype="int8")
    for (s, e), lab in zip(slices, seq_labels):
        if lab == -1:
            row_flags[s:e] = -1
    return row_flags


def ground_truth_series(y_seq: np.ndarray, normal_label: str) -> dd.Series:
    """Wrap sequence labels back into a dask series shaped like the benchmark summary expects."""
    series = pd.Series(np.where(y_seq == 1, "Anomaly", normal_label))
    return dd.from_pandas(series, npartitions=1)


class Playbook(ABC):
    """Template-method base for all four concrete playbooks.

    Subclasses set LABELED/OUTPUT_MODE class attributes and implement the
    abstract hooks. Concrete `run()` drives the per-strategy loop identically
    for every subclass.
    """

    LABELED: ClassVar[bool]
    OUTPUT_MODE: ClassVar[Literal["w", "a"]] = "w"
    VOTE_THRESHOLD: ClassVar[float] = 0.5

    def __init__(self, connector: Optional[DataConnector] = None):
        self._connector = connector
        self._y_col: Optional[str] = None
        self._normal_label: Optional[str] = None

    # ---- concrete / final ----

    def build_dags(self) -> Tuple[dd.DataFrame, Dict[str, Dict], Dict[str, Any]]:
        cfg = os.environ.get("CONFIG_LABELS")
        if self.LABELED and not cfg:
            logger.error(
                "%s requires CONFIG_LABELS (e.g. 'class:Normal').",
                type(self).__name__,
            )
            sys.exit(1)
        if not self.LABELED and cfg:
            logger.error(
                "%s must not set CONFIG_LABELS (got %r).",
                type(self).__name__, cfg,
            )
            sys.exit(1)

        if self.LABELED:
            parts = cfg.split(":", 1)
            if len(parts) != 2:
                logger.error(
                    "CONFIG_LABELS must be 'column:normal_value' (got %r).", cfg,
                )
                sys.exit(1)
            self._y_col, self._normal_label = parts[0], parts[1]

        builder = TransformerBuilder(overrides=OVERRIDES)
        required_cols = builder.required_columns(self._y_col)

        if self._connector is None:
            self._connector = self._make_connector()
        ddf = self._connector.load(usecols=required_cols)

        jobs = builder.build_all(
            scoring_fn=SCORING if self.LABELED else None,
            ddf=ddf,
            expand_grids=not self.LABELED,
        )

        total_tasks = sum(
            len(j["feature_pipes"]) * (len(j["pipelines"]) + len(j["searches"]))
            for j in jobs.values()
        )
        build_info: Dict[str, Any] = {
            "connector": type(self._connector).__name__,
            "data_source": (
                getattr(self._connector, "path", None)
                or getattr(self._connector, "index", None)
            ),
            "config_path": os.getenv("CONFIG_PATH"),
            "required_columns": required_cols,
            "strategies": list(jobs.keys()),
            "total_tasks": total_tasks,
            "labeled": self.LABELED,
            "y_config": [self._y_col, self._normal_label] if self.LABELED else None,
        }
        return ddf, jobs, build_info

    def run(self, client: Client) -> Dict[str, List[Dict[str, Any]]]:
        ddf, jobs, build_info = self.build_dags()
        logger.info(
            "%s build info:\n%s",
            type(self).__name__,
            json.dumps(build_info, indent=2, default=str),
        )

        ddf = client.persist(ddf)
        full_df = ddf.compute()

        ground_truth: Optional[dd.Series] = None
        if self.LABELED and self._y_col is not None:
            ground_truth = ddf[self._y_col]

        results_path, target_path = self._reset_output_files()
        result_connector = CSVConnector()

        all_results: Dict[str, List[Dict[str, Any]]] = {}

        for strategy_name, job in jobs.items():
            feature_pipes = job["feature_pipes"]
            pipelines = job["pipelines"]
            searches = job["searches"]

            n_tasks = len(feature_pipes) * (len(pipelines) + len(searches))
            logger.info(
                "=== Strategy: %s  (%d feature pipes x (%d pipelines + %d searches) = %d tasks) ===",
                strategy_name, len(feature_pipes), len(pipelines), len(searches), n_tasks,
            )

            if not feature_pipes or (not pipelines and not searches):
                logger.warning(
                    "Strategy '%s' has no feature pipes or no estimators, skipping.",
                    strategy_name,
                )
                continue

            strategy_results: List[Dict[str, Any]] = []

            for feat_desc, col_pipes in feature_pipes:
                logger.debug("  Fitting composite feature pipe: %s", feat_desc)
                t0 = time.perf_counter()
                X_full = self._fit_feature_pipes(client, ddf, col_pipes)
                elapsed = time.perf_counter() - t0
                logger.info(
                    "  Composite feature '%s' computed in %.1fs  shape=%s",
                    feat_desc, elapsed, getattr(X_full, "shape", "?"),
                )

                n_rows = len(full_df)
                X_model, ctx = self._prepare_X(client, X_full, feat_desc, n_rows)
                if X_model is None:
                    logger.warning(
                        "  _prepare_X returned None for '%s'; skipping feature pipe.",
                        feat_desc,
                    )
                    continue

                Y_model = self._prepare_Y(client, ground_truth, ctx) if self.LABELED else None

                group_labels: Dict[str, List[np.ndarray]] = defaultdict(list)

                tasks: List[Tuple[str, Any, str]] = []
                for entry in pipelines:
                    desc, pipe, group_key = entry
                    tasks.append((desc, pipe, group_key))
                for s_desc, search in searches:
                    tasks.append((s_desc, search, s_desc))

                for desc, est_obj, group_key in tasks:
                    task_name = f"{strategy_name}__{feat_desc}__{desc}"
                    logger.debug("    Fitting: %s", task_name)
                    t0 = time.perf_counter()
                    trained_obj, labels = self._fit_and_label(
                        est_obj, X_model, Y_model, client,
                    )
                    client.run(gc.collect)
                    elapsed = time.perf_counter() - t0
                    logger.info("    Completed in %.1fs: %s", elapsed, task_name)

                    summary = self._summarize(
                        task_name, strategy_name, trained_obj, labels,
                        ground_truth, ctx,
                    )
                    strategy_results.append(summary)
                    result_connector.save_row(summary)
                    self._log_task_summary(task_name, summary)

                    self._after_task(
                        client=client,
                        strategy=strategy_name,
                        feat_desc=feat_desc,
                        task_name=task_name,
                        trained_obj=trained_obj,
                        labels=labels,
                        ctx=ctx,
                        full_df=full_df,
                        target_path=target_path,
                    )

                    group_labels[group_key].append(np.asarray(labels))

                for group_key, per_model_labels in group_labels.items():
                    vote_summary = self._finalize_group(
                        client=client,
                        strategy=strategy_name,
                        feat_desc=feat_desc,
                        group_key=group_key,
                        per_model_labels=per_model_labels,
                        ctx=ctx,
                        full_df=full_df,
                        target_path=target_path,
                    )
                    if vote_summary is not None:
                        strategy_results.append(vote_summary)
                        result_connector.save_row(vote_summary)

                del X_full, X_model
                client.run(gc.collect)
                try:
                    client.run(trim_memory)
                except OSError:
                    pass
                gc.collect()

            logger.debug("Strategy '%s' complete.\n", strategy_name)
            all_results[strategy_name] = strategy_results

        return all_results

    def _fit_feature_pipes(
        self,
        client: Client,
        ddf: dd.DataFrame,
        col_pipes: Dict[str, Any],
    ) -> da.Array:
        col_arrays = []
        for col_name, pipe in col_pipes.items():
            col = ddf[col_name]
            if pd.api.types.is_numeric_dtype(col.dtype):
                col = col.fillna(0)
            else:
                col = col.fillna("").astype(str)
            col_series = client.persist(col.to_bag())
            feat = pipe.fit_transform(col_series)
            if not isinstance(feat, da.Array):
                if hasattr(feat, "to_dask_array"):
                    feat = feat.to_dask_array(lengths=True)
                else:
                    feat = da.from_array(np.asarray(feat, dtype="float32"))
            feat = feat.persist()
            if any(np.isnan(c) for c in feat.chunks[0]):
                feat.compute_chunk_sizes()
            if feat.ndim == 1:
                feat = feat.reshape(-1, 1)
            col_arrays.append(feat)

        row_chunks = col_arrays[0].chunks[0]
        col_arrays = [a.rechunk({0: row_chunks, 1: -1}) for a in col_arrays]
        X_dask = client.persist(da.concatenate(col_arrays, axis=1))
        return X_dask

    def _reset_output_files(self) -> Tuple[Optional[str], Optional[str]]:
        results_path = os.environ.get("RESULTS_PATH")
        if results_path and os.path.exists(results_path):
            os.remove(results_path)
        target_path = os.environ.get("TARGET_PATH")
        if self.OUTPUT_MODE == "a" and target_path and os.path.exists(target_path):
            os.remove(target_path)
        return results_path, target_path

    def _write_anomaly_rows(
        self,
        full_df: pd.DataFrame,
        mask: np.ndarray,
        target_path: str,
    ) -> int:
        anomaly_rows = full_df[mask]
        if self.OUTPUT_MODE == "w":
            anomaly_rows.to_csv(target_path, index=False, sep=";", mode="w")
        else:
            header = not os.path.exists(target_path)
            anomaly_rows.to_csv(
                target_path, index=False, sep=";", mode="a", header=header,
            )
        return len(anomaly_rows)

    def _log_task_summary(self, task_name: str, summary: Dict[str, Any]) -> None:
        if self.LABELED:
            logger.info(
                "  %s  n_anomalies=%d  accuracy=%.4f  precision=%.4f  recall=%.4f  f1=%.4f",
                task_name,
                summary.get("n_anomalies", 0),
                summary.get("accuracy", 0.0),
                summary.get("precision", 0.0),
                summary.get("recall", 0.0),
                summary.get("f1_score", 0.0),
            )
        else:
            logger.info(
                "  %s  n_samples=%d  n_anomalies=%d",
                task_name,
                summary.get("n_samples", 0),
                summary.get("n_anomalies", 0),
            )

    # ---- hooks with defaults ----

    def _after_task(
        self,
        client: Client,
        strategy: str,
        feat_desc: str,
        task_name: str,
        trained_obj: Any,
        labels: np.ndarray,
        ctx: FeatureContext,
        full_df: pd.DataFrame,
        target_path: Optional[str],
    ) -> None:
        """Default: no side-effect after a per-task summary. Benchmarks override."""
        return None

    def _prepare_Y(
        self,
        client: Client,
        ground_truth: Optional[dd.Series],
        ctx: FeatureContext,
    ) -> Optional[da.Array]:
        """Default: no Y. Labeled subclasses override."""
        return None

    # ---- abstract ----

    @abstractmethod
    def _make_connector(self) -> DataConnector: ...

    @abstractmethod
    def _prepare_X(
        self,
        client: Client,
        X_dask: da.Array,
        feat_desc: str,
        n_rows: int,
    ) -> Tuple[Any, FeatureContext]: ...

    @abstractmethod
    def _fit_and_label(
        self,
        est_obj: Any,
        X: Any,
        Y: Optional[da.Array],
        client: Client,
    ) -> Tuple[Any, np.ndarray]: ...

    @abstractmethod
    def _summarize(
        self,
        task_name: str,
        strategy: str,
        trained_obj: Any,
        labels: np.ndarray,
        ground_truth: Optional[dd.Series],
        ctx: FeatureContext,
    ) -> Dict[str, Any]: ...

    @abstractmethod
    def _expand_row_mask(
        self,
        labels: np.ndarray,
        ctx: FeatureContext,
        n_rows: int,
    ) -> np.ndarray: ...

    @abstractmethod
    def _finalize_group(
        self,
        client: Client,
        strategy: str,
        feat_desc: str,
        group_key: str,
        per_model_labels: List[np.ndarray],
        ctx: FeatureContext,
        full_df: pd.DataFrame,
        target_path: Optional[str],
    ) -> Optional[Dict[str, Any]]: ...
