from time import sleep
from dotenv import load_dotenv
from dask import config as dask_config
from dask.distributed import LocalCluster
from pipeline.playbook_1 import run as run_playbook_1


def main():
    load_dotenv()
    dask_config.refresh()
    cluster = LocalCluster(
    n_workers=2,
    threads_per_worker=2,
    )
    client = cluster.get_client()
    print(client)
    sleep(10)  # Give the cluster time to start up
    try:
        run_playbook_1(client=client)
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
