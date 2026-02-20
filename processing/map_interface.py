from abc import ABC, abstractmethod
from collections import Counter
import math
import re
from typing import Callable, List, Optional, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import zlib
import numpy as np
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
    def tfidf_vectorize(
        series: dd.Series,
        column: Optional[str] = None,
        max_features: int = 100,
        token_pattern: str = r"\b\w+\b",
        lowercase: bool = False,
        min_df: int = 2,
        max_df: float = 0.5,
        sublinear_tf: bool = True,
    ) -> dd.DataFrame:
        series = series.fillna("").astype(str)
        prefix = f"{column}__" if column else ""
        columns = [f"{prefix}{i}" for i in range(max_features)]

        partitions = series.to_delayed()

        @delayed
        def _fit(parts):
            full = pd.concat(parts)
            vec = TfidfVectorizer(
                max_features=max_features,
                token_pattern=token_pattern,
                lowercase=lowercase,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=sublinear_tf,
            )
            vec.fit(full)
            return vec

        @delayed
        def _transform(part, vec):
            arr = vec.transform(part).toarray()
            return pd.DataFrame(arr, index=part.index, columns=columns, dtype='float64')

        fitted = _fit(partitions)
        transformed = [_transform(p, fitted) for p in partitions]

        meta = pd.DataFrame({c: pd.Series(dtype='float64') for c in columns})
        return dd.from_delayed(transformed, meta=meta)

    @staticmethod
    def word_ngram_vectorize(
        series: dd.Series,
        column: Optional[str] = None,
        max_features: int = 100,
        token_pattern: str = r"\b\w+\b",
        lowercase: bool = False,
        min_df: int = 2,
        max_df: float = 0.5,
        ngram_range: tuple = (2, 2),
    ) -> dd.DataFrame:
        """Bag of word n-grams. Default is bigrams; set ngram_range=(3,3) for trigrams, etc."""
        series = series.fillna("").astype(str)
        prefix = f"{column}__" if column else ""
        columns = [f"{prefix}{i}" for i in range(max_features)]

        partitions = series.to_delayed()

        @delayed
        def _fit(parts):
            full = pd.concat(parts)
            vec = CountVectorizer(
                max_features=max_features,
                token_pattern=token_pattern,
                ngram_range=tuple(ngram_range),
                analyzer="word",
                lowercase=lowercase,
                min_df=min_df,
                max_df=max_df,
            )
            vec.fit(full)
            return vec

        @delayed
        def _transform(part, vec):
            arr = vec.transform(part).toarray()
            return pd.DataFrame(arr, index=part.index, columns=columns, dtype='int64')

        fitted = _fit(partitions)
        transformed = [_transform(p, fitted) for p in partitions]

        meta = pd.DataFrame({c: pd.Series(dtype='int64') for c in columns})
        return dd.from_delayed(transformed, meta=meta)

    @staticmethod
    def char_vectorize(
        series: dd.Series,
        column: Optional[str] = None,
        max_features: int = 100,
        ngram_range: tuple = (2, 4),
        analyzer: str = "char_wb",
        lowercase: bool = False,
        min_df: int = 2,
        max_df: float = 0.5,
    ) -> dd.DataFrame:
        series = series.fillna("").astype(str)
        prefix = f"{column}__" if column else ""
        columns = [f"{prefix}{i}" for i in range(max_features)]

        partitions = series.to_delayed()

        @delayed
        def _fit(parts):
            full = pd.concat(parts)
            vec = CountVectorizer(
                max_features=max_features,
                ngram_range=tuple(ngram_range),
                analyzer=analyzer,
                lowercase=lowercase,
                min_df=min_df,
                max_df=max_df,
            )
            vec.fit(full)
            return vec

        @delayed
        def _transform(part, vec):
            arr = vec.transform(part).toarray()
            return pd.DataFrame(arr, index=part.index, columns=columns, dtype='int64')

        fitted = _fit(partitions)
        transformed = [_transform(p, fitted) for p in partitions]

        meta = pd.DataFrame({c: pd.Series(dtype='int64') for c in columns})
        return dd.from_delayed(transformed, meta=meta)

    @staticmethod
    def frequency_encode(
        series: dd.Series,
        column: str,
        normalize: bool = False,
        max_features: int = 20,
        token_pattern: str = r"[^\s/\\]+",
        min_df: int = 2,
        max_df: float = 0.5,
    ) -> dd.DataFrame:
        """Positional IDF encoding via sklearn TfidfVectorizer.

        Tokenizes each value with *token_pattern*, computes IDF across all
        samples, then writes the IDF of the token at position *i* into column
        *i* (zero-padded when fewer tokens).  When *normalize* is True each
        column is min-max scaled to [0, 1].
        """
        series = series.fillna("").astype(str)
        prefix = f"{column}__" if column else ""
        columns = [f"{prefix}{i}" for i in range(max_features)]
        partitions = series.to_delayed()

        @delayed
        def _fit_idf(parts, pattern):
            full = pd.concat(parts)
            vec = TfidfVectorizer(
                token_pattern=pattern,
                use_idf=True,
                smooth_idf=True,
                norm=None,
                max_features=None,  # vocabulary not limited — we need IDF for every token
                min_df=min_df,
                max_df=max_df,
            )
            vec.fit(full)
            # map token -> idf weight
            return dict(zip(vec.get_feature_names_out(), vec.idf_))

        @delayed
        def _transform(part, idf_map, pattern, n_features, do_normalize):
            pat = re.compile(pattern)
            rows = []
            for text in part:
                tokens = pat.findall(text)
                vec = [idf_map.get(tokens[i], 0.0) if i < len(tokens) else 0.0
                       for i in range(n_features)]
                rows.append(vec)
            arr = np.array(rows, dtype='float64')
            if do_normalize:
                col_min = arr.min(axis=0)
                col_max = arr.max(axis=0)
                span = col_max - col_min
                span[span == 0] = 1.0
                arr = (arr - col_min) / span
            return pd.DataFrame(arr, index=part.index, columns=columns)

        idf_map = _fit_idf(partitions, token_pattern)
        transformed = [
            _transform(p, idf_map, token_pattern, max_features, normalize)
            for p in partitions
        ]

        meta = pd.DataFrame({c: pd.Series(dtype='float64') for c in columns})
        return dd.from_delayed(transformed, meta=meta)

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
