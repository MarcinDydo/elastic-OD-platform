from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Union
import json
import os
import pandas as pd
import dask.dataframe as dd
from elasticsearch import Elasticsearch, helpers 
#RequestsHttpConnection  

from .connector_interface import DataConnector


def _flatten_dict(data: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    items = []
    for key, value in data.items():
        new_key = f"{parent}{sep}{key}" if parent else key
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


#https://docs.dask.org/en/latest/dataframe-create.html#mapping-from-a-function -  delayed df from map 

@dataclass
class ElasticConnector(DataConnector):
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    index: Optional[str] = None
    query: Optional[Union[str, Dict[str, Any]]] = None
    use_ssl: bool = True
    verify_certs: bool = False
    max_docs: Optional[int] = None
    npartitions: int = 4
    scan_size: int = 1000
    name: str = "elastic"
    dataframe: Optional[dd.DataFrame] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.host is None:
            self.host = os.getenv("ELASTIC_IP")
        if self.port is None:
            self.port = int(os.getenv("ELASTIC_PORT", "9200"))
        if self.api_key is None:
            self.api_key = os.getenv("ELASTIC_API_KEY")
        if self.index is None:
            self.index = os.getenv("ELASTIC_INDEX")
        if self.query is None:
            query_path = os.getenv("ELASTIC_QUERY_PATH")
            if query_path:
                with open(query_path, "r", encoding="utf-8") as fh:
                    self.query = json.load(fh)

    def _client(self) -> Elasticsearch:
        scheme = "https" if self.use_ssl else "http"
        return Elasticsearch(
            [{"host": self.host, "port": self.port, "scheme": "https"}],
            api_key=self.api_key,
            verify_certs=False,
            timeout=30,
            #connection_class=RequestsHttpConnection,
        )

    def _query_dict(self) -> Dict[str, Any]:
        if self.query is None:
            raise RuntimeError(
                "No Elasticsearch query provided (set query or ELASTIC_QUERY_PATH)."
            )
        if isinstance(self.query, str):
            return json.loads(self.query)
        return self.query

    def _iter_rows(self) -> Iterable[Dict[str, Any]]:
        client = self._client()
        payload = self._query_dict()
        scan_iter = helpers.scan(
            client=client,
            index=self.index,
            query=payload,
            size=self.scan_size,
            preserve_order=False,
        )
        for idx, hit in enumerate(scan_iter):
            if self.max_docs and idx >= self.max_docs:
                break
            source = hit.get("_source", {})
            row = _flatten_dict(source)
            row["_id"] = hit.get("_id")
            row["_index"] = hit.get("_index")
            yield row

    def load(self, usecols: Optional[Sequence[str]] = None) -> dd.DataFrame:
        rows = list(self._iter_rows())
        if not rows:
            self.dataframe = dd.from_pandas(pd.DataFrame(), npartitions=1)
            return self.dataframe
        pdf = pd.DataFrame(rows)
        if usecols is not None:
            missing = [col for col in usecols if col not in pdf.columns]
            if missing:
                raise ValueError(f"Requested columns not available in elastic data: {missing}")
        self.dataframe = dd.from_pandas(pdf, npartitions=self.npartitions)
        return self.dataframe
