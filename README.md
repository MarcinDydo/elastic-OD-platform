## Platforma eksperymentalna do porównawczej oceny nienadzorowanych metod detekcji anomalii w heterogenicznych logach systemów informatycznych.   

Struktura projektu:

main.py - punkt wejściowy i konfiguracja clienta Dask.
processing/playbook_*.py - receptura uruchamiania jobów na klastrze.
configs/config.json - opis konkretnych jobów
pipeline/*.py - pomocnicze pliki źródłowe

## Konfiguracja środowiska:
Na windowsie przy pomocy "wsl":

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Uruchamianie
w katalogu root projektu powinien znajdowac sie plik .env:

```
CSV_PATH = "./data/csic-01.csv"
CSV_ASSUME_MISSING = "True"
CSV_BLOCKSIZE = "64MB"
CSV_RESULT_PATH = "./data/csic-01-experiment3-results.csv"
CONFIG_PATH = "./configs/config.json"
MALLOC_TRIM_THRESHOLD_ = "0"
DASK_DISTRIBUTED__WORKER__MEMORY__LIMIT = "10GiB"
LOG_LEVEL = "DEBUG"
# DASK_DATAFRAME__BACKEND= "cudf"
DASK_DISTRIBUTED__WORKER__MEMORY__SPILL = "0.7"
DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE = "0.9"
DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE = "0.95"
ELASTIC_IP = ""
ELASTIC_API_KEY = ""  
ELASTIC_QUERY_PATH = ""
```
następnie można uruchomić program poleceniem:

```
.venv/bin/python3 main.py
```

## Troubleshooting
można patrzeć jak działa program pod adresem:
```
http://127.0.0.1:8787/status
```
