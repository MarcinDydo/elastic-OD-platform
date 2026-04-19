"""Unlabeled per-row outlier detection (Elastic source, majority voting across grid combos)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client

from connectors.connector_interface import DataConnector
from connectors.elastic_connector import ElasticConnector
from processing.playbook import FeatureContext, Playbook, task_summary, vote_labels

logger = logging.getLogger(__name__)


class PointPlaybook(Playbook):
    LABELED = False
    OUTPUT_MODE = "a"

    def _make_connector(self) -> DataConnector:
        return ElasticConnector()

    def _prepare_X(
        self,
        client: Client,
        X_dask: da.Array,
        feat_desc: str,
        n_rows: int,
    ) -> Tuple[Any, FeatureContext]:
        return X_dask, FeatureContext(feat_desc=feat_desc, n_rows=n_rows)

    def _fit_and_label(
        self,
        est_obj: Any,
        X: Any,
        Y: Optional[da.Array],
        client: Client,
    ) -> Tuple[Any, np.ndarray]:
        est_obj.fit(X)
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
        return task_summary(task_name, strategy, labels)

    def _expand_row_mask(
        self,
        labels: np.ndarray,
        ctx: FeatureContext,
        n_rows: int,
    ) -> np.ndarray:
        return np.asarray(labels)

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
        if not per_model_labels:
            return None

        if len(per_model_labels) == 1:
            # Singleton group — write the single model's anomalies, no vote summary.
            if target_path:
                mask = np.asarray(per_model_labels[0]) == -1
                if mask.any():
                    n_written = self._write_anomaly_rows(full_df, mask, target_path)
                    logger.info(
                        "    Appended %d anomaly rows to %s", n_written, target_path,
                    )
            return None

        voted = vote_labels(per_model_labels, self.VOTE_THRESHOLD)
        vote_task = f"{strategy}__{feat_desc}__{group_key}__voted"
        summary = task_summary(
            vote_task, strategy, voted, extra={"n_models": len(per_model_labels)},
        )
        logger.info(
            "    %s  n_models=%d  n_anomalies=%d (majority vote)",
            vote_task, len(per_model_labels), summary["n_anomalies"],
        )

        if target_path and summary["n_anomalies"] > 0:
            n_written = self._write_anomaly_rows(full_df, voted == -1, target_path)
            logger.info(
                "    Appended %d voted anomaly rows to %s", n_written, target_path,
            )
        return summary
