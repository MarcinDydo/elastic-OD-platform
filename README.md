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
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
source .venv/bin/activate
```

## Uruchamianie
w katalogu root projektu powinien znajdowac sie plik .env:

```
CSV_PATH = "./data/csic-01.csv"
CSV_ASSUME_MISSING = "True"
CSV_BLOCKSIZE = "64MB"

PLAYBOOK = "sequence"
TARGET_PATH = "./output/csic_experiment_anomalies.csv"
RESULTS_PATH = "./output/csic_experiment_results.csv"

CONFIG_PATH = "./configs/csic_config.json"
CONFIG_LABELS = "class:Normal"
LOG_LEVEL = "INFO"

MALLOC_TRIM_THRESHOLD_ = "0"
DASK_DISTRIBUTED__WORKER__MEMORY__LIMIT = "10GiB"
# DASK_DATAFRAME__BACKEND= "cudf"
DASK_DISTRIBUTED__WORKER__MEMORY__SPILL = "0.7"
DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE = "0.9"
DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE = "0.95"

ELASTIC_IP = ""
ELASTIC_API_KEY = ""  
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
