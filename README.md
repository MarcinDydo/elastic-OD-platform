## Platforma eksperymentalna do porównawczej oceny nienadzorowanych metod detekcji anomalii w heterogenicznych logach systemów informatycznych.   

Struktura projektu:

main.py - punkt wejściowy i konfiguracja clienta Dask.
env.py - konfiguracja środowiska
processing/playbook_*.py - receptura uruchamiania jobów.
configs/config.json - opis konkretnych jobów
pipeline/*.py - pomocnicze pliki źródłowe

## Konfiguracja środowiska:
Na windowsie przy pomocy "wsl":

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

## Uruchamianie
w katalogu root projektu powinien znajdowac sie plik .env:

```
PLAYBOOK = "elastic_point"
WORKERS = 1
THREADS_PER_WORKER = 2
TARGET_PATH = "./output/suricata_experiment_anomalies.csv"
RESULTS_PATH = "./output/suricata_experiment_results.csv"

CONFIG_PATH = "./configs/elastic_config.json"
#CONFIG_LABELS = "label:norm"
LOG_LEVEL = "INFO"

MALLOC_TRIM_THRESHOLD_ = "0"
DASK_DISTRIBUTED__WORKER__MEMORY__LIMIT = "12GiB"
# DASK_DATAFRAME__BACKEND= "cudf"
DASK_DISTRIBUTED__WORKER__MEMORY__SPILL = "0.7"
DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE = "0.9"
DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE = "0.95"

CSV_PATH = "./data/HttpParams_full.csv"
CSV_ASSUME_MISSING = "True"
CSV_BLOCKSIZE = "64MB"

ELASTIC_IP = "127.0.0.1"
ELASTIC_PORT = 19200
ELASTIC_API_KEY = ""   
ELASTIC_INDEX = "logs-*"
ELASTIC_QUERY_PATH = "./configs/elastic_query.json"
```


Aby następnie pobrać próbki i wyznaczyć odstające dane można uruchomić program poleceniem:

```
.venv/bin/python3 main.py
```

## Troubleshooting
można patrzeć jak działa program pod adresem:
```
http://127.0.0.1:8787/status
```
