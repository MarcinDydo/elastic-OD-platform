from time import sleep
import logging
import os


from dotenv import load_dotenv
from dask import config as dask_config
from dask.distributed import Client, LocalCluster

logger = logging.getLogger(__name__)


def _resolve_playbook():
    name = os.getenv("PLAYBOOK", "point").lower()
    if name == "sequence":
        from processing.sequence_benchmark import run
    else:
        from processing.point_benchmark import run
    return run


def main():
    dask_config.refresh()
    run_playbook = _resolve_playbook()

    if os.getenv("DASK_DATAFRAME__BACKEND") == "cudf":
        from dask_cuda import LocalCUDACluster
        dask_config.set({"dataframe.backend": "cudf"})
        cluster = LocalCUDACluster()
        client = Client(cluster)
    else:
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=2,
            memory_limit=os.getenv("DASK_DISTRIBUTED__WORKER__MEMORY__LIMIT", "12GB"),
        )
        client = cluster.get_client()

    logger.info("Dask cluster ready: %s", client.dashboard_link)
    sleep(10)  # Allow time for dashboard to initialize
    try:
        run_playbook(client=client)
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
