import importlib
import logging
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from sklearn.model_selection import ParameterGrid
from dask_ml.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from pipeline.transformers.utils import FillNaTransformer

logger = logging.getLogger(__name__)


class TransformerBuilder:

    def __init__(
        self,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, type]] = None,
    ):
        self.config = self._load_config(config_path)
        self.overrides: Dict[str, type] = overrides or {}

    @staticmethod
    def _load_config(path: Optional[str]) -> dict:
        config_path = path or os.getenv("CONFIG_PATH")
        if not config_path:
            raise RuntimeError("CONFIG_PATH is required for builder configuration.")
        config_file = Path(config_path)
        if not config_file.exists():
            raise RuntimeError(f"Builder config not found: {config_file}")
        with config_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("Builder config JSON must be an object.")
        return data

    @staticmethod
    def _expand_grid(grid: Dict[str, list]) -> List[Dict[str, Any]]:
        if not grid:
            return [{}]
        combos = list(ParameterGrid(grid))
        for combo in combos:
            for key, val in combo.items():
                if isinstance(val, list):
                    combo[key] = tuple(val)
        return combos

    def _resolve_class(self, dotted_path: str) -> type:
        if dotted_path in self.overrides:
            return self.overrides[dotted_path]
        module_path, _, class_name = dotted_path.rpartition(".")
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @staticmethod
    def _maybe_wrap_pyod(cls, params):
        """If *cls* is a PyOD BaseDetector, return a PyODDetectorWrapper."""
        try:
            from pyod.models.base import BaseDetector
            if isinstance(cls, type) and issubclass(cls, BaseDetector):
                from pipeline.transformers.wrappers import PyODDetectorWrapper
                return PyODDetectorWrapper(pyod_cls=cls, **params)
        except ImportError:
            pass
        return cls(**params)

    def _build_feature_pipes(
        self,
        features_spec: Dict[str, Dict[str, Dict]],
    ) -> List[Tuple[str, Dict[str, Pipeline]]]:
        results: List[Tuple[str, Dict[str, Pipeline]]] = []

        for col_name, callables_dict in features_spec.items():
            chain_items = list(callables_dict.items())
            # Start with FillNa as the first step
            combos: List[Tuple[List[str], List[tuple]]] = [([], [("fillna", FillNaTransformer())])]

            for callable_path, call_spec in chain_items:
                static_params = dict(call_spec.get("params", {}))
                grid = call_spec.get("grid", {})
                grid_combos = self._expand_grid(grid)
                cls = self._resolve_class(callable_path)
                step_name = cls.__name__

                new_combos = []
                for desc_parts, steps_so_far in combos:
                    for grid_params in grid_combos:
                        merged = {**static_params, **grid_params}
                        tag = ""
                        if grid_params:
                            tag = "_".join([f"{k}={v}" for k, v in sorted(grid_params.items())])
                        desc = desc_parts + [f"{step_name}({tag})" if tag else step_name]
                        instance = cls(**merged)
                        step_label = f"{step_name}_{len(steps_so_far)}"
                        new_steps = steps_so_far + [(step_label, instance)]
                        new_combos.append((desc, new_steps))

                combos = new_combos

            for desc_parts, steps in combos:
                desc_str = f"{col_name}__{'__'.join(desc_parts)}"
                pipe = Pipeline(steps)
                logger.debug("Feature pipe: %s", desc_str)
                results.append((desc_str, {col_name: pipe}))

        logger.debug("Built %d feature pipes.", len(results))
        return results


    def _build_estimator_objects(
        self,
        estimators_spec: Dict[str, List[Dict]],
        scoring_fn,
    ) -> Tuple[List[Tuple[str, Pipeline]], List[Tuple[str, GridSearchCV]]]:
        pipelines: List[Tuple[str, Pipeline]] = []
        searches: List[Tuple[str, GridSearchCV]] = []

        for est_name, steps_list in estimators_spec.items():
            pipeline_steps = []
            param_grid: Dict[str, list] = {}
            has_grid = False

            for idx, step_def in enumerate(steps_list):
                callable_path = step_def["callable"]
                static_params = dict(step_def.get("params", {}))
                grid = step_def.get("grid", {})
                threshold_spec = step_def.get("threshold", {})

                cls = self._resolve_class(callable_path)
                step_label = f"{est_name}_{idx}" if len(steps_list) > 1 else est_name

                for param_name, thresh_path in threshold_spec.items():
                    thresh_cls = self._resolve_class(thresh_path)
                    thresh_instance = thresh_cls()
                    logger.debug(
                        "Threshold override: %s.%s = %r",
                        est_name, param_name, thresh_instance,
                    )
                    static_params[param_name] = thresh_instance
                    grid.pop(param_name, None)

                instance = self._maybe_wrap_pyod(cls, static_params)
                pipeline_steps.append((step_label, instance))

                if grid:
                    has_grid = True
                    for param_key, param_values in grid.items():
                        full_key = f"{step_label}__{param_key}"
                        param_grid[full_key] = param_values

            pipe = Pipeline(pipeline_steps)

            if has_grid:
                search = GridSearchCV(
                    pipe,
                    param_grid,
                    scoring=scoring_fn,
                    refit=True,
                )
                desc = f"{est_name}__grid"
                logger.debug("GridSearchCV: %s  grid=%s", desc, param_grid)
                searches.append((desc, search))
            else:
                desc = est_name
                logger.debug("Pipeline: %s", desc)
                pipelines.append((desc, pipe))

        logger.debug(
            "Built %d pipelines and %d grid searches.",
            len(pipelines),
            len(searches),
        )
        return pipelines, searches

    # Public API - core functionality
    def build_transformer(
        self,
        name: str,
        spec: Dict[str, Any],
        scoring_fn,
    ) -> Dict[str, Any]:
        features_spec = spec.get("features", {})
        estimators_spec = spec.get("estimators", {})

        feature_pipes = self._build_feature_pipes(features_spec)
        pipelines, searches = self._build_estimator_objects(estimators_spec, scoring_fn)

        total = len(feature_pipes) * (len(pipelines) + len(searches))
        logger.info(
            "Strategy '%s': %d feature pipes x (%d pipelines + %d searches) = %d tasks",
            name, len(feature_pipes), len(pipelines), len(searches), total,
        )
        return {
            "feature_pipes": feature_pipes,
            "pipelines": pipelines,
            "searches": searches,
        }

    def required_columns(self,y_name="class") -> List[str]:
        cols: set = set()
        for spec in self.config.values():
            cols.update(spec.get("features", {}).keys())
        cols.add(y_name)
        return sorted(cols)

    def build_all(
        self,
        scoring_fn=None,
    ) -> Dict[str, Dict[str, Any]]:
        strategies: Dict[str, Dict[str, Any]] = {}
        for name, spec in self.config.items():
            logger.debug("Building pipelines for strategy: %s", name)
            strategies[name] = self.build_transformer(name, spec, scoring_fn)
        return strategies
