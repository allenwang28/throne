from pathlib import Path

import pytest

pytest.importorskip("monarch.job")

from throne.cluster import Cluster
from monarch.job import SlurmJob


def test_slurm_metadata_written(tmp_path: Path):
    job = SlurmJob(
        meshes={"pool": 2},
        partition="debug",
        time_limit="01:00:00",
        gpus_per_node=4,
        ntasks_per_node=1,
        python_exe="python3",
        job_name="throne-test",
        monarch_port=23456,
        log_dir="/tmp/slurm-logs",
        exclusive=False,
        cpus_per_task=4,
        mem="32G",
        slurm_args=("--qos=normal", "--account=test"),
    )
    job._slurm_job_id = "12345"

    cluster = Cluster(job, name="slurm_cluster", state_dir=tmp_path)

    cluster.save()

    metadata_path = tmp_path / "slurm_cluster" / "metadata.txt"
    metadata = Cluster._parse_metadata(metadata_path)

    assert metadata["Job Type"] == "SlurmJob"
    assert metadata["Partition"] == "debug"
    assert metadata["Job ID"] == "12345"
    assert metadata["GPUs per node"] == "4"
    assert metadata["Nodes"] == "2"
    assert "slurm_cluster/job_state.pkl" in metadata["State Path"]
    assert "SBATCH --partition=debug" in metadata["SBATCH"]
