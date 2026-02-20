import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dask.dataframe as dd
from dask.delayed import delayed
from processing.map_interface import MapInterface

def _load_config(
    path: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    config_path = path or os.getenv("MAPPER_CONFIG_PATH")
    if not config_path:
        raise RuntimeError("MAPPER_CONFIG_PATH is required for mapper configuration.")
    config_file = Path(config_path)
    if not config_file.exists():
        raise RuntimeError(f"Mapper config not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Mapper config JSON must be an object.")
    return data

def _apply_strategy(
    series: dd.Series,
    strategy: str,
    strategy_params: Dict[str, Any],
) -> Optional[dd.DataFrame]:
    feature_name = f"{series.name}_{strategy}"
    
    match strategy:
        case "none" | "passthrough":
            return series.to_frame(name=feature_name)
        case "drop":
            return None
        case "standardize":
            return MapInterface.standardize(series, column=feature_name)
        case "minmax":
            return series.pipe(MapInterface.minmax).to_frame(name=feature_name)
        case "count_vectorizer":
            params = strategy_params.get(strategy, {})
            return MapInterface.count_vectorize(
                series,
                column=feature_name,
                max_features=params["max_features"],
                token_pattern=params["token_pattern"],
                lowercase=params["lowercase"],
            )  # returns delayed DataFrame /list of dealyed object from map_partitions 
        case "ngram_vectorizer":
            params = strategy_params.get(strategy, {})
            return MapInterface.word_ngram_vectorize(
                series,
                column=feature_name,
                max_features=params.get("max_features", 100),
                token_pattern=params.get("token_pattern", r"\b\w+\b"),
                lowercase=params.get("lowercase", False),
                min_df=params.get("min_df", 2),
                max_df=params.get("max_df", 0.5),
                ngram_range=tuple(params.get("ngram_range", [2, 2])),
            )
        case "frequency_encode":
            params = strategy_params.get(strategy, {})
            return MapInterface.frequency_encode(
                series,
                column=feature_name,
                normalize=params.get("normalize", False),
                max_features=params.get("max_features", 20),
                token_pattern=params.get("token_pattern", r"[^\s/\\]+"),
                min_df=params.get("min_df", 2),
                max_df=params.get("max_df", 0.5),
            )
        case "binary_encode":
            params = strategy_params.get(strategy, {})
            return MapInterface.binary_encode(
                series, column=feature_name, drop_original=params["drop_original"],
                bits=params["bits"], hash_seed=params["hash_seed"],
            )  # Already returns DataFrame
        case "tfidf":
            params = strategy_params.get(strategy, {})
            return MapInterface.tfidf_vectorize(
                series,
                column=feature_name,
                max_features=params.get("max_features", 100),
                token_pattern=params.get("token_pattern", r"\b\w+\b"),
                lowercase=params.get("lowercase", False),
                min_df=params.get("min_df", 2),
                max_df=params.get("max_df", 0.5),
                sublinear_tf=params.get("sublinear_tf", True),
            )
        case "char_vectorizer":
            params = strategy_params.get(strategy, {})
            return MapInterface.char_vectorize(
                series,
                column=feature_name,
                max_features=params.get("max_features", 100),
                ngram_range=tuple(params.get("ngram_range", [2, 4])),
                analyzer=params.get("analyzer", "char_wb"),
                lowercase=params.get("lowercase", False),
                min_df=params.get("min_df", 2),
                max_df=params.get("max_df", 0.5),
            )
        case _:
            raise ValueError(f"Unknown strategy: {strategy}")

class TransformerMapper(MapInterface):
    def __init__(
        self,
        config_path: Optional[str] = None,
        encodings: List[Tuple[Any]] = None
    ):
        self.encodings = encodings or []
        strategy_params = _load_config(config_path)
        self.strategy_params = strategy_params or {}
        self.result = None

    def check_params(self) -> bool: #TODO:
        # if self.params.
        return True

    def map(self, ddf: dd.DataFrame) -> Dict[str, dd.DataFrame]:    
        self.result = {
            f"{col}_{strat}": _apply_strategy(ddf[col], strat, self.strategy_params) # "str": reference to delayed objects
            for col, strat in self.encodings
            if col in ddf.columns
        }
        return self.result
