from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import dask.dataframe as dd

class ReduceInterface(ABC):
    """Reduces a Dask DataFrame into derived artifacts (models, metrics, etc.)."""

    name: str
    features: Optional[List[str]] = None
    params: Optional[Dict[str, Any]] = None

    @abstractmethod
    def reduce(self, ddf: dd.DataFrame):
        ...

    @abstractmethod
    def check_params(self) -> bool:
        ...

    def prepare(self, lddf: Dict[str, dd.DataFrame]) -> dd.DataFrame:   
        selected = [lddf[feature] for feature in self.features if feature in lddf]
        
        if not selected:
            raise ValueError(f"No features found for reducer '{self.name}': {self.features}")
        
        if len(selected) == 1:
            return selected[0]
        
        return dd.concat(selected, axis=1)


    @staticmethod
    def _resolve_config_path(path: Optional[str]) -> Path:
        env_path = os.getenv("REDUCE_CONFIG_PATH")
        if not path and not env_path:
            raise RuntimeError("REDUCE_CONFIG_PATH is required for reducer configuration.")
        return Path(path) if path else Path(env_path)

    @staticmethod
    def load_config(path: Optional[str] = None) -> Dict:
        config_path = ReduceInterface._resolve_config_path(path)
        if not config_path.exists():
            raise RuntimeError(f"Reducer config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Reducer config JSON must be an object.")
        return data
