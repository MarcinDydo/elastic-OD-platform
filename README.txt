`wsl`


sudo apt-get install python3
sudo apt-get install graphviz


`cd /mnt/e/School/Sem2mgr/magistera`

`python3 -m venv .venv`

`source .venv/bin/activate`

`pip install "apache-airflow==3.1.3" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.1.3/constraints-3.13.txt"` 

`pip install pyarrow pandas`
#nie
`pip install apache-airflow-providers-jdbc`

`cp /mnt/e/School/Sem2mgr/magistera/src/*.py ./dags/`



```
curl -v -k --url 'https://10.44.30.201:9200/_cat/indices' --request GET -H 'Authorization: ApiKey x1cSjqDFRrKubNQguo079w'
-H 'Authorization: ApiKey x1cSjqDFRrKubNQguo079w'
curl -v -k -H "Authorization: ApiKey dmM2emxaVUJ5Q3ZOSGxhcDI0SEI6eDFjU2pxREZSckt1Yk5RZ3VvMDc5dw==" https://10.44.30.201:9200/_cat/indices
```


http://127.0.0.1:8787/status
