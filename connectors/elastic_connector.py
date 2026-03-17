from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Union
import json
import pandas as pd
import dask.dataframe as dd
from elasticsearch import Elasticsearch, RequestsHttpConnection, helpers
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
    host: str
    port: int
    api_key: str
    index: str
    query: Union[str, Dict[str, Any]]
    use_ssl: bool = True
    verify_certs: bool = False
    max_docs: Optional[int] = None
    npartitions: int = 4
    name: str = "elastic"
    dataframe: Optional[dd.DataFrame] = field(default=None, init=False, repr=False)

    def _client(self) -> Elasticsearch:
        return Elasticsearch(
            [{"host": self.host, "port": self.port}],
            connection_class=RequestsHttpConnection,
            api_key=self.api_key,
            use_ssl=self.use_ssl,
            verify_certs=self.verify_certs,
            timeout=30,
        )

    def _query_dict(self) -> Dict[str, Any]:
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
            pdf = pdf[list(usecols)]
        self.dataframe = dd.from_pandas(pdf, npartitions=self.npartitions)
        return self.dataframe
