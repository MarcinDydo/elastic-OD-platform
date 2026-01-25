from abc import ABC, abstractmethod
from collections import Counter
import math
import re
from typing import Callable, List, Optional
from sklearn.feature_extraction.text import CountVectorizer
import zlib
import dask.dataframe as dd
import pandas as pd
from dask.delayed import delayed

StrategyFunc = Callable[[dd.DataFrame], dd.DataFrame]


class MapInterface(ABC):
    """Transforms a Dask DataFrame into feature frames."""

    name: str

    @abstractmethod
    def map(self, ddf: dd.DataFrame) -> List[dd.DataFrame]:
        ...

    @staticmethod
    def standardize(series: dd.Series, column: str) -> dd.DataFrame:
        partitions = series.to_delayed()
        
        @delayed
        def _compute_stats(parts):
            full = pd.concat(parts)
            numeric = pd.to_numeric(full, errors="coerce").astype("float64")
            return numeric.mean(), numeric.std()
        
        @delayed
        def _transform(part, stats):
            mean, std = stats
            std = std if std != 0 else 1
            numeric = pd.to_numeric(part, errors="coerce").astype("float64")
            scaled = ((numeric - mean) / std).fillna(0)
            return scaled.to_frame(name=column)
        
        stats = _compute_stats(partitions)
        transformed = [_transform(p, stats) for p in partitions]
        
        meta = pd.DataFrame({column: pd.Series(dtype='float64')})
        return dd.from_delayed(transformed, meta=meta)  #lazy dask 

    @staticmethod
    def minmax(series: dd.Series, min_val: Optional[float] = None, max_val: Optional[float] = None) -> dd.Series:
        series = dd.to_numeric(series, errors="coerce")
        if min_val is None or max_val is None:
            min_val = series.min()
            max_val = series.max()
        span = max_val - min_val
        span = span + (span == 0)
        scaled = (series.astype("float64") - min_val) / span
        return scaled.fillna(0)
    
    @staticmethod
    def count_vectorize(
        series: dd.Series,
        column: Optional[str] = None,
        max_features: int = 100,
        token_pattern: str = r"\b\w+\b",
        lowercase: bool = False,
        min_df: int = 2,
        max_df: float = 0.5,
    ) -> dd.DataFrame:
        series = series.fillna("").astype(str)
        prefix = f"{column}__" if column else ""
        columns = [f"{prefix}{i}" for i in range(max_features)]
        
        partitions = series.to_delayed()  # List of delayed pd.Series

        # 1. Fit once on all partitions combined
        @delayed
        def _fit(parts):
            full = pd.concat(parts)
            vec = CountVectorizer(
                max_features=max_features,
                token_pattern=token_pattern,
                lowercase=lowercase,
                min_df=min_df,
                max_df=max_df,
            )
            vec.fit(full)
            #print(vec)
            return vec

        # 2. Transform single partition (runs in parallel per partition)
        @delayed
        def _transform(part, vec):
            arr = vec.transform(part).toarray()
            return pd.DataFrame(arr, index=part.index, columns=columns, dtype='int64')

        fitted = _fit(partitions)  # Delayed vectorizer (fit once)
        transformed = [_transform(p, fitted) for p in partitions]  # Parallel transforms

        meta = pd.DataFrame({c: pd.Series(dtype='int64') for c in columns})
        return dd.from_delayed(transformed, meta=meta) #delayed dataframes / sparse matrix

    @staticmethod
    def frequency_encode(series: dd.Series, column: str, normalize: bool = False) -> dd.DataFrame:
        partitions = series.to_delayed()
        
        @delayed
        def _compute_freqs(parts, normalize):
            full = pd.concat(parts)
            counts = full.value_counts()
            if normalize:
                counts = counts / counts.sum()
            return counts
        
        @delayed
        def _transform(part, freqs):
            mapped = part.map(freqs).astype("float64").fillna(0)
            return mapped.to_frame(name=column)
        
        freqs = _compute_freqs(partitions, normalize)
        transformed = [_transform(p, freqs) for p in partitions]
        
        meta = pd.DataFrame({column: pd.Series(dtype='float64')})
        return dd.from_delayed(transformed, meta=meta) # Return DataFrame, not Series

    @staticmethod
    def binary_encode(
        series: dd.Series,
        column: str,
        drop_original: bool = True,
        bits: Optional[int] = None,
        hash_seed: Optional[int] = None,
    ) -> dd.DataFrame:
        n_bits = max(1, int(bits) if bits is not None else 16)
        hash_key: Optional[str]
        if hash_seed is None:
            hash_key = None
        else:
            # pandas expects a 16-character string key
            hash_key = f"{int(hash_seed) & 0xFFFFFFFFFFFFFFFF:016x}"

        def encode_partition(part):
            hashed = pd.util.hash_pandas_object(
                part.fillna("").astype(str),
                index=False,
                hash_key=hash_key,
            ).to_numpy(dtype="uint64", copy=False)
            data = {
                f"{column}__bin{bit}": ((hashed >> bit) & 1).astype("int8")
                for bit in range(n_bits)
            }
            return pd.DataFrame(data, index=part.index)

        meta = pd.DataFrame({f"{column}__bin{bit}": pd.Series(dtype="int8") for bit in range(n_bits)})
        encoded = series.map_partitions(encode_partition, meta=meta)
        if drop_original:
            return encoded
        # keep the original as well
        return dd.concat([series.to_frame(name=column), encoded], axis=1)

    # @staticmethod
    # def text_stats(series: dd.Series, column: str) -> dd.DataFrame:
    #     def token_entropy(token: str) -> float:
    #         counts = Counter(token)
    #         length = len(token)
    #         if not length:
    #             return 0.0
    #         entropy = 0.0
    #         for count in counts.values():
    #             p = count / length
    #             entropy -= p * math.log2(p)
    #         return entropy

    #     def compute_partition(part: pd.Series) -> pd.DataFrame:
    #         rows = []
    #         for value in part:
    #             if pd.isna(value):
    #                 text = ""
    #             else:
    #                 text = str(value)
    #             stripped = "".join(ch for ch in text if not ch.isspace())
    #             length = len(stripped)
    #             denom = length if length else 1
    #             upper = sum(1 for ch in stripped if ch.isupper())
    #             lower = sum(1 for ch in stripped if ch.islower())
    #             digits = sum(1 for ch in stripped if ch.isdigit())
    #             punct = sum(1 for ch in stripped if ch in ".,;:!?")
    #             symbols = sum(
    #                 1 for ch in stripped if not ch.isalnum() and ch not in ".,;:!?"
    #             )
    #             tokens = re.findall(r"[A-Za-z0-9]+", text)
    #             tokens = [token for token in tokens if not token.isdigit()]
    #             if tokens:
    #                 mean_entropy = sum(token_entropy(token) for token in tokens) / len(tokens)
    #             else:
    #                 mean_entropy = 0.0
    #             if text:
    #                 raw = text.encode("utf-8")
    #                 compression_ratio = len(zlib.compress(raw)) / len(raw)
    #             else:
    #                 compression_ratio = 0.0
    #             rows.append(
    #                 {
    #                     f"{column}__upper_ratio": upper / denom,
    #                     f"{column}__lower_ratio": lower / denom,
    #                     f"{column}__digit_ratio": digits / denom,
    #                     f"{column}__punct_ratio": punct / denom,
    #                     f"{column}__symbol_ratio": symbols / denom,
    #                     f"{column}__mean_token_entropy": mean_entropy,
    #                     f"{column}__compression_ratio": compression_ratio,
    #                 }
    #             )
    #         return pd.DataFrame(rows, index=part.index)

    #     meta = pd.DataFrame(
    #         {
    #             f"{column}__upper_ratio": pd.Series(dtype="float64"),
    #             f"{column}__lower_ratio": pd.Series(dtype="float64"),
    #             f"{column}__digit_ratio": pd.Series(dtype="float64"),
    #             f"{column}__punct_ratio": pd.Series(dtype="float64"),
    #             f"{column}__symbol_ratio": pd.Series(dtype="float64"),
    #             f"{column}__mean_token_entropy": pd.Series(dtype="float64"),
    #             f"{column}__compression_ratio": pd.Series(dtype="float64"),
    #         } #TODO:this should be a dask dataframe 
    #     )
    #     return series.map_partitions(compute_partition, meta=meta)
