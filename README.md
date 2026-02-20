## Platforma eksperymentalna do porównawczej oceny nienadzorowanych metod detekcji anomalii w heterogenicznych logach systemów informatycznych.   

obecnie program ma jedną scieżkę, nie przyjmuje argumentów.
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python3 main.py
```
w katalogu root projektu powinien znajdowac sie plik .env:

```
IP = ""
API_KEY = ""  
CSV_PATH = "./data/csic-01.csv"
CSV_ASSUME_MISSING = "True"
CSV_BLOCKSIZE = "64MB"
GRAPH_PATH = "./data/csic-01-nowy.jpg"
CSV_RESULT_PATH = "./data/csic-01-experiment-results.csv"
MAPPER_CONFIG_PATH = "./configs/mapper_config.json"
REDUCE_CONFIG_PATH = "./configs/reduce_config.json"
DASK_DISTRIBUTED__WORKER__MEMORY__TARGET = "0.6"
DASK_DISTRIBUTED__WORKER__MEMORY__LIMIT = "8GB"
DASK_DISTRIBUTED__WORKER__MEMORY__SPILL = "0.7"
```

można patrzeć jak działa program pod adresem:
```
http://127.0.0.1:8787/status
```