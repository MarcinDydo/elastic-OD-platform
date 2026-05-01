"""Labeled windowed outlier detection benchmark (CSV source, metrics vs ground truth).
Aggregates consecutive feature vectors into fixed-size windows (mean or sum),
runs detection on the windowed array, then expands sequence-level flags back
to row level for TARGET_PATH output.
"""
from __future__ import annotations

import json
import logging
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client
from dask_ml.model_selection import GridSearchCV

from connectors.connector_interface import DataConnector
from connectors.csv_connector import CSVConnector
from processing.playbook import (
    FeatureContext, Playbook, build_summary,
    WINDOW_DEFAULTS,
    aggregate_features,
    aggregate_labels,
    expand_labels_to_rows,
    ground_truth_series,
)

logger = logging.getLogger(__name__)


class SequenceBenchmark(Playbook):
    LABELED = True
    OUTPUT_MODE = "w"
    WINDOW: ClassVar[Dict[str, Any]] = WINDOW_DEFAULTS

    def _make_connector(self) -> DataConnector:
        return CSVConnector()

    def _prepare_X(
        self,
        client: Client,
        X_dask: da.Array,
        feat_desc: str,
        n_rows: int,
    ) -> Tuple[Any, FeatureContext]:
        X_combined = X_dask.compute()
        if hasattr(X_combined, "toarray"):
            X_combined = X_combined.toarray()
        X_combined = np.asarray(X_combined, dtype="float32")

        X_seq, slices = aggregate_features(
            X_combined,
            size=self.WINDOW["size"],
            overlap=self.WINDOW["overlap"],
            method=self.WINDOW["method"],
        )
        logger.info(
            "  '%s' aggregated rows %d -> sequences %d  shape=%s",
            feat_desc, X_combined.shape[0], X_seq.shape[0], X_seq.shape,
        )
        if X_seq.shape[0] == 0:
            logger.warning(
                "  No sequences produced (n_rows<%d). Skipping feature pipe.",
                self.WINDOW["size"],
            )
            return None, FeatureContext(feat_desc=feat_desc, n_rows=n_rows)

        X_seq_dask = client.persist(da.from_array(X_seq, chunks="64MB"))
        return X_seq_dask, FeatureContext(
            feat_desc=feat_desc, n_rows=n_rows, slices=slices,
        )

    def _prepare_Y(
        self,
        client: Client,
        ground_truth: Optional[dd.Series],
        ctx: FeatureContext,
    ) -> Optional[da.Array]:
        if ground_truth is None or ctx.slices is None:
            return None
        y_row = (ground_truth != self._normal_label).astype("int8")
        y_np = y_row.to_dask_array(lengths=True).compute()
        y_seq = aggregate_labels(y_np, ctx.slices)
        ctx.extra["y_seq"] = y_seq
        ctx.extra["seq_ground_truth"] = ground_truth_series(y_seq, self._normal_label)
        return client.persist(da.from_array(y_seq, chunks="auto"))

    def _fit_and_label(
        self,
        est_obj: Any,
        X: Any,
        Y: Optional[da.Array],
        client: Client,
    ) -> Tuple[Any, np.ndarray]:
        if isinstance(est_obj, GridSearchCV):
            est_obj.fit(X, y=Y)
            best_est = est_obj.best_estimator_.steps[-1][1]
            best_est.labels_ = best_est.predict_chunked(X)
            return est_obj, best_est.labels_
        est_obj.fit(X, y=Y)
        est = est_obj.steps[-1][1]
        est.labels_ = est.predict_chunked(X)
        return est_obj, est.labels_

    def _summarize(
        self,
        task_name: str,
        strategy: str,
        trained_obj: Any,
        labels: np.ndarray,
        ground_truth: Optional[dd.Series],
        ctx: FeatureContext,
    ) -> Dict[str, Any]:
        seq_ground_truth = ctx.extra.get("seq_ground_truth")
        row_flags = self._expand_row_mask(labels, ctx, ctx.n_rows)
        extra: Dict[str, Any] = {
            "sequence_config": json.dumps(self.WINDOW),
            "n_row_anomalies": int((row_flags == -1).sum()),
        }
        return build_summary(
            task_name=task_name,
            strategy=strategy,
            trained_obj=trained_obj,
            labels=labels,
            ground_truth=seq_ground_truth,
            normal_label=self._normal_label,
            extra=extra,
        )

    def _expand_row_mask(
        self,
        labels: np.ndarray,
        ctx: FeatureContext,
        n_rows: int,
    ) -> np.ndarray:
        if ctx.slices is None:
            return np.asarray(labels)
        return expand_labels_to_rows(np.asarray(labels), ctx.slices, n_rows)

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
        if not target_path:
            return
        row_flags = self._expand_row_mask(labels, ctx, len(full_df))
        n_written = self._write_anomaly_rows(full_df, row_flags == -1, target_path)
        logger.info(
            "    Wrote %d anomaly rows (%d flagged sequences) to %s",
            n_written, int((np.asarray(labels) == -1).sum()), target_path,
        )

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
    ) -> Optional[Dict[str, Any]]:
        return None
