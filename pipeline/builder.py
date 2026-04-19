import importlib
import logging
import json
import numbers
import os
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import dask
import torch
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from dask_ml.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from pipeline.transformations.utils import (
    DenseTransformer,
    NanToNumDaskTransformer,
)

logger = logging.getLogger(__name__)


# (name, position, factory) keyed by canonical dotted path.
# Auto-inserted by the builder when the corresponding class appears in the
# feature chain so the config can reference the native dask_ml classes.
_AUTO_STEPS: Dict[str, List[Tuple[str, str, Any]]] = {
    "dask_ml.feature_extraction.text.CountVectorizer": [
        ("densify", "after", DenseTransformer()),
    ],
    "dask_ml.feature_extraction.text.HashingVectorizer": [
        ("densify", "after", DenseTransformer()),
    ],
    "dask_ml.decomposition.TruncatedSVD": [
        ("nan_to_num", "before", NanToNumDaskTransformer()),
    ],
}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _count_vocab_dask(series, cv_kwargs: Dict[str, Any]) -> Dict[str, int]:
    """Global vocabulary for a dask Series, matching CountVectorizer semantics.
    """
    from sklearn.feature_extraction.text import CountVectorizer as SklearnCountVectorizer

    analyzer_kwargs = {
        k: v for k, v in cv_kwargs.items()
        if k not in ("max_features", "max_df", "min_df", "vocabulary")
    }
    analyzer = SklearnCountVectorizer(**analyzer_kwargs).build_analyzer()

    max_features = cv_kwargs.get("max_features")
    max_df = cv_kwargs.get("max_df", 1.0)
    min_df = cv_kwargs.get("min_df", 1)

    def _partition_tfdf(part):
        tf = Counter()
        df = Counter()
        n_docs = 0
        for doc in part:
            n_docs += 1
            feature_counter: Dict[str, int] = {}
            for feature in analyzer(doc):
                if feature in feature_counter:
                    feature_counter[feature] += 1
                else:
                    feature_counter[feature] = 1
            for feat, cnt in feature_counter.items():
                tf[feat] += cnt
                df[feat] += 1
        return tf, df, n_docs

    col = series.fillna("").astype(str)
    parts = col.to_delayed()
    results = dask.compute(*[dask.delayed(_partition_tfdf)(p) for p in parts])

    tf_total: Counter = Counter()
    df_total: Counter = Counter()
    n_docs = 0
    for tf, df, nd in results:
        tf_total.update(tf)
        df_total.update(df)
        n_docs += nd

    if n_docs == 0 or not tf_total:
        raise ValueError(
            "empty vocabulary; perhaps the documents only contain stop words"
        )

    max_doc_count = (
        max_df if isinstance(max_df, numbers.Integral)
        else int(round(max_df * n_docs))
    )
    min_doc_count = (
        min_df if isinstance(min_df, numbers.Integral)
        else int(round(min_df * n_docs))
    )
    if max_doc_count < min_doc_count:
        raise ValueError("max_df corresponds to fewer documents than min_df")

    surviving = [
        t for t in tf_total
        if min_doc_count <= df_total[t] <= max_doc_count
    ]
    if not surviving:
        raise ValueError(
            "After pruning, no terms remain. Try a lower min_df or higher max_df."
        )

    if max_features is not None and len(surviving) > max_features:
        surviving.sort(key=lambda t: (-tf_total[t], t))
        surviving = surviving[:max_features]

    surviving.sort()
    return {t: i for i, t in enumerate(surviving)}


def _inspect_countvectorizer(
    col_name: str,
    params: Dict[str, Any],
    ddf,
    cache: Dict[Any, Dict[str, int]],
) -> Dict[str, Any]:
    """Inject a global vocabulary into CountVectorizer params when max_features is set."""
    max_features = params.get("max_features")
    if max_features is None:
        return params
    if "vocabulary" in params and params["vocabulary"] is not None:
        return params
    if ddf is None:
        logger.debug(
            "CountVectorizer inspector: ddf unavailable for %r, skipping vocab precomputation",
            col_name,
        )
        return params

    cv_kwargs = {k: v for k, v in params.items() if k != "vocabulary"}
    cache_key = (col_name, _freeze(cv_kwargs))
    if cache_key in cache:
        vocab = cache[cache_key]
    else:
        logger.info(
            "Precomputing CountVectorizer vocabulary for %r (max_features=%s)",
            col_name, max_features,
        )
        vocab = _count_vocab_dask(ddf[col_name], cv_kwargs)
        cache[cache_key] = vocab
        logger.debug("Vocabulary for %r: %d tokens", col_name, len(vocab))

    new_params = dict(params)
    new_params["vocabulary"] = vocab
    return new_params


_PARAM_INSPECTORS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "dask_ml.feature_extraction.text.CountVectorizer": _inspect_countvectorizer,
}


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
        TransformerBuilder._normalize_tuple_params(data)
        return data

    _TUPLE_PARAMS = frozenset({"ngram_range"})

    @staticmethod
    def _normalize_tuple_params(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in TransformerBuilder._TUPLE_PARAMS and isinstance(value, list):
                    node[key] = tuple(value)
                else:
                    TransformerBuilder._normalize_tuple_params(value)
        elif isinstance(node, list):
            for item in node:
                TransformerBuilder._normalize_tuple_params(item)

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
                from pipeline.transformations.wrappers import PyODDetectorWrapper
                return PyODDetectorWrapper(pyod_class=cls, **params)
        except ImportError:
            pass
        return cls(**params)

    def _build_feature_pipes(
        self,
        features_spec: Dict[str, Dict[str, Dict]],
        ddf=None,
    ) -> List[Tuple[str, Dict[str, Pipeline]]]:
        # Phase 1: build per-column variant lists
        per_column: Dict[str, List[Tuple[str, Pipeline]]] = {}
        vocab_cache: Dict[Any, Dict[str, int]] = {}

        for col_name, callables_dict in features_spec.items():
            chain_items = list(callables_dict.items())
            combos: List[Tuple[List[str], List[tuple]]] = [([], [])]

            for callable_path, call_spec in chain_items:
                static_params = dict(call_spec.get("params", {}))
                grid = call_spec.get("grid", {})
                grid_combos = self._expand_grid(grid)
                cls = self._resolve_class(callable_path)
                step_name = cls.__name__

                auto_steps = _AUTO_STEPS.get(callable_path, [])
                for a_name, a_pos, _ in auto_steps:
                    logger.debug(
                        "Auto-inserting %r step %s %s",
                        a_name, a_pos, step_name,
                    )

                inspector = _PARAM_INSPECTORS.get(callable_path)

                new_combos = []
                for desc_parts, steps_so_far in combos:
                    for grid_params in grid_combos:
                        merged = {**static_params, **grid_params}
                        if inspector is not None:
                            merged = inspector(col_name, merged, ddf, vocab_cache)
                        tag = ""
                        if grid_params:
                            tag = "_".join([f"{k}={v}" for k, v in sorted(grid_params.items())])
                        desc = desc_parts + [f"{step_name}({tag})" if tag else step_name]
                        instance = cls(**merged)
                        step_label = f"{step_name}_{len(steps_so_far)}"
                        chain = list(steps_so_far)
                        for a_name, a_pos, a_cls in auto_steps:
                            if a_pos == "before":
                                chain.append((f"{a_name}_{len(chain)}", a_cls))
                        chain.append((step_label, instance))
                        for a_name, a_pos, a_cls in auto_steps:
                            if a_pos == "after":
                                chain.append((f"{a_name}_{len(chain)}", a_cls))
                        new_combos.append((desc, chain))

                combos = new_combos

            col_variants = []
            for desc_parts, steps in combos:
                desc_str = f"{col_name}__{'__'.join(desc_parts)}"
                pipe = Pipeline(steps)
                logger.debug("Feature pipe: %s", desc_str)
                col_variants.append((desc_str, pipe))
            per_column[col_name] = col_variants

        # Phase 2: cross-product across all columns
        col_names = list(per_column.keys())
        col_variant_lists = [per_column[c] for c in col_names]

        results: List[Tuple[str, Dict[str, Pipeline]]] = []
        for combo in product(*col_variant_lists):
            descs = [item[0] for item in combo]
            combined_desc = " + ".join(descs)
            col_pipes = {
                col_names[i]: combo[i][1]
                for i in range(len(col_names))
            }
            results.append((combined_desc, col_pipes))

        logger.debug("Built %d composite feature pipes.", len(results))
        return results


    def _build_estimator_objects(
        self,
        estimators_spec: Dict[str, List[Dict]],
        scoring_fn,
        expand_grids: bool = False,
    ) -> Tuple[List[Tuple[str, Pipeline, str]], List[Tuple[str, GridSearchCV]]]:
        pipelines: List[Tuple[str, Pipeline, str]] = []
        searches: List[Tuple[str, GridSearchCV]] = []

        for est_name, steps_list in estimators_spec.items():
            pipeline_steps = []
            param_grid: Dict[str, list] = {}
            has_grid = False

            for idx, step_def in enumerate(steps_list):
                callable_path = step_def["class"]
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
                if expand_grids:
                    for combo in ParameterGrid(param_grid):
                        cloned = clone(pipe)
                        cloned.set_params(**combo)
                        tag = "_".join(
                            f"{k.split('__')[-1]}={v}"
                            for k, v in sorted(combo.items())
                        )
                        desc = f"{est_name}__{tag}" if tag else est_name
                        logger.debug("Pipeline (expanded): %s", desc)
                        pipelines.append((desc, cloned, est_name))
                else:
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
                pipelines.append((desc, pipe, est_name))

        logger.debug(
            "Built %d pipelines and %d grid searches.",
            len(pipelines),
            len(searches),
        )
        return pipelines, searches

    # Public API - core functionality
    def build_transformations(
        self,
        name: str,
        spec: Dict[str, Any],
        scoring_fn,
        ddf=None,
        expand_grids: bool = False,
    ) -> Dict[str, Any]:
        features_spec = spec.get("features", {})
        estimators_spec = spec.get("estimators", {})

        feature_pipes = self._build_feature_pipes(features_spec, ddf=ddf)
        pipelines, searches = self._build_estimator_objects(
            estimators_spec, scoring_fn, expand_grids=expand_grids,
        )

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

    def required_columns(self, y_name: Optional[str] = "class") -> List[str]:
        cols: set = set()
        for spec in self.config.values():
            cols.update(spec.get("features", {}).keys())
        if y_name:
            cols.add(y_name)
        return sorted(cols)

    def build_all(
        self,
        scoring_fn=None,
        ddf=None,
        expand_grids: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        strategies: Dict[str, Dict[str, Any]] = {}
        for name, spec in self.config.items():
            logger.debug("Building pipelines for strategy: %s", name)
            strategies[name] = self.build_transformations(
                name, spec, scoring_fn, ddf=ddf, expand_grids=expand_grids,
            )
        return strategies
