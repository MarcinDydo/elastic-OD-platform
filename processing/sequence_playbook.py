"""
Unlabeled windowed outlier detection (Elasticsearch source, majority voting across grid combos).
Aggregates consecutive feature vectors into fixed-size windows (mean or sum),
runs detection on the windowed array, votes across per-combo variants, and
expands voted sequence flags back to row level for TARGET_PATH.
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

from connectors.connector_interface import DataConnector
from connectors.elastic_connector import ElasticConnector
from processing.playbook import (
    FeatureContext, Playbook, task_summary, vote_labels,
    WINDOW_DEFAULTS,
    aggregate_features,
    expand_labels_to_rows,
)

logger = logging.getLogger(__name__)


class SequencePlaybook(Playbook):
    LABELED = False
    OUTPUT_MODE = "a"
    WINDOW: ClassVar[Dict[str, Any]] = WINDOW_DEFAULTS

    def _make_connector(self) -> DataConnector:
        return ElasticConnector()

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
        labels = np.asarray(labels)
        row_flags = self._expand_row_mask(labels, ctx, ctx.n_rows)
        extra: Dict[str, Any] = {
            "sequence_config": json.dumps(self.WINDOW),
            "n_sequences": int(labels.shape[0]),
            "n_seq_anomalies": int((labels == -1).sum()),
            "n_rows": ctx.n_rows,
            "n_row_anomalies": int((row_flags == -1).sum()),
        }
        return task_summary(task_name, strategy, labels, extra=extra)

    def _expand_row_mask(
        self,
        labels: np.ndarray,
        ctx: FeatureContext,
        n_rows: int,
    ) -> np.ndarray:
        if ctx.slices is None:
            return np.asarray(labels)
        return expand_labels_to_rows(np.asarray(labels), ctx.slices, n_rows)

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
            if target_path:
                row_flags = self._expand_row_mask(
                    per_model_labels[0], ctx, len(full_df),
                )
                mask = row_flags == -1
                if mask.any():
                    n_written = self._write_anomaly_rows(full_df, mask, target_path)
                    logger.info(
                        "    Appended %d anomaly rows to %s", n_written, target_path,
                    )
            return None

        voted = vote_labels(per_model_labels, self.VOTE_THRESHOLD)
        row_flags = self._expand_row_mask(voted, ctx, len(full_df))
        n_seq_anomalies = int((voted == -1).sum())
        n_row_anomalies = int((row_flags == -1).sum())

        vote_task = f"{strategy}__{feat_desc}__{group_key}__voted"
        summary = task_summary(
            vote_task, strategy, voted,
            extra={
                "n_models": len(per_model_labels),
                "n_seq_anomalies": n_seq_anomalies,
                "n_row_anomalies": n_row_anomalies,
                "sequence_config": json.dumps(self.WINDOW),
            },
        )
        logger.info(
            "    %s  n_models=%d  n_seq_anomalies=%d  n_row_anomalies=%d (vote)",
            vote_task, len(per_model_labels), n_seq_anomalies, n_row_anomalies,
        )

        if target_path and n_row_anomalies > 0:
            n_written = self._write_anomaly_rows(full_df, row_flags == -1, target_path)
            logger.info(
                "    Appended %d voted anomaly rows to %s", n_written, target_path,
            )
        return summary
