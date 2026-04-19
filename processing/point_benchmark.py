"""Labeled per-row outlier detection benchmark (CSV source, metrics vs ground truth)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client
from dask_ml.model_selection import GridSearchCV

from connectors.connector_interface import DataConnector
from connectors.csv_connector import CSVConnector
from processing.playbook import FeatureContext, Playbook, build_summary

logger = logging.getLogger(__name__)


class PointBenchmark(Playbook):
    LABELED = True
    OUTPUT_MODE = "w"

    def _make_connector(self) -> DataConnector:
        return CSVConnector()

    def _prepare_X(
        self,
        client: Client,
        X_dask: da.Array,
        feat_desc: str,
        n_rows: int,
    ) -> Tuple[Any, FeatureContext]:
        return X_dask, FeatureContext(feat_desc=feat_desc, n_rows=n_rows)

    def _prepare_Y(
        self,
        client: Client,
        ground_truth: Optional[dd.Series],
        ctx: FeatureContext,
    ) -> Optional[da.Array]:
        if ground_truth is None:
            return None
        Y_dask = (ground_truth != self._normal_label).astype("int8").to_dask_array(lengths=True)
        return client.persist(Y_dask)

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
        return build_summary(
            task_name=task_name,
            strategy=strategy,
            trained_obj=trained_obj,
            labels=labels,
            ground_truth=ground_truth,
            normal_label=self._normal_label,
        )

    def _expand_row_mask(
        self,
        labels: np.ndarray,
        ctx: FeatureContext,
        n_rows: int,
    ) -> np.ndarray:
        return np.asarray(labels)

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
        row_labels = self._expand_row_mask(labels, ctx, len(full_df))
        n_written = self._write_anomaly_rows(full_df, row_labels == -1, target_path)
        logger.info("    Wrote %d anomaly rows to %s", n_written, target_path)

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
