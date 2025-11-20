"""SLURM cluster definition example."""
from monarch.job import SlurmJob

job = SlurmJob(
    meshes={"pool": 4},  # Request 4 nodes
    gpus_per_node=8,
    time_limit="24:00:00",
    # Optional: partition, account, etc.
    # partition="gpu",
    # account="my_account",
)
