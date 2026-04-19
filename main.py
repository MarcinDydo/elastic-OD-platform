from time import sleep
import logging
import os


from dotenv import load_dotenv
from dask import config as dask_config
from dask.distributed import Client, LocalCluster

logger = logging.getLogger(__name__)


def _resolve_playbook_cls():
    name = os.getenv("PLAYBOOK", "point").lower()
    from processing.point_benchmark import PointBenchmark
    from processing.sequence_benchmark import SequenceBenchmark
    from processing.point_playbook import PointPlaybook
    from processing.sequence_playbook import SequencePlaybook
    return {
        "point": PointBenchmark,
        "sequence": SequenceBenchmark,
        "elastic_point": PointPlaybook,
        "elastic_sequence": SequencePlaybook,
    }.get(name, PointBenchmark)


def main():
    dask_config.set({"dataframe.query-planning": False})
    dask_config.set({"distributed.dashboard.bokeh-application.session_token_expiration": 86400000})
    dask_config.refresh()
    playbook_cls = _resolve_playbook_cls()

    if os.getenv("DASK_DATAFRAME__BACKEND") == "cudf":
        from dask_cuda import LocalCUDACluster
        dask_config.set({"dataframe.backend": "cudf"})
        cluster = LocalCUDACluster()
        client = Client(cluster)
    else:
        cluster = LocalCluster(
            n_workers=int(os.getenv("WORKERS","1")),
            threads_per_worker=int(os.getenv("THREADS_PER_WORKER","2")),
            memory_limit=os.getenv("DASK_DISTRIBUTED__WORKER__MEMORY__LIMIT", "12GB"),
        )
        client = cluster.get_client()

    logger.info("Dask cluster ready: %s", client.dashboard_link)
    sleep(10)  # Allow time for dashboard to initialize
    try:
        playbook_cls().run(client=client)
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
