"""SLURM cluster definition example (explicit params)."""

from monarch.job import SlurmJob
from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from throne.spec import ClusterSpec

class SlurmSpec(ClusterSpec):
    """Optional helper to bundle Slurm job with controller/worker env."""

    def __init__(
        self,
        job,
        controller_env=None,
        worker_env=None,
        transport: ChannelTransport = ChannelTransport.TcpWithHostname,
    ):
        self.job = job
        self._controller_env = controller_env or {}
        self._worker_env = worker_env or {}
        self._transport = transport

    def get_job(self):
        return self.job

    @property
    def controller_env(self):
        env = dict(self._controller_env)
        env.setdefault("HYPERACTOR_DEFAULT_TRANSPORT", self._transport.value)
        return env

    @property
    def worker_env(self):
        env = dict(self._worker_env)
        env.setdefault("HYPERACTOR_DEFAULT_TRANSPORT", self._transport.value)
        return env

    @property
    def configure_kwargs(self):
        return {"default_transport": self._transport}

# Edit these to match your SLURM cluster. If unsure, run `sinfo -s` to see partitions.
job = SlurmJob(
    meshes={"pool": 4},  # total nodes per mesh
    gpus_per_node=4,
    ntasks_per_node=1,
    partition="batch",  # change to a valid partition for your cluster
    time_limit="24:00:00",
    job_name="throne-slurm-demo",
    log_dir="./slurm_logs",
    slurm_args=(
        "--qos=normal",
        # "--account=your_account",
        # "--constraint=A100",
    ),
    # Optional: cpus_per_task=4, mem="0", exclusive=False
)

spec = SlurmSpec(
    job,
    controller_env={},
    worker_env={},
)
