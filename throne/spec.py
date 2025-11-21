"""
ClusterSpec hooks to carry cluster-specific setup logic.

A spec exposes:
- get_job(): returns the underlying JobTrait
- configure(): ensures monarch_hyperactor is configured for the cluster
- on_init(): optional hook to run before applying the job (e.g., set env)
- controller_env/worker_env: optional dicts with environment variables that
  throne can use to configure controller and worker processes.
"""

from typing import Any, Dict

from monarch.job import JobTrait
from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from monarch._rust_bindings.monarch_hyperactor.config import configure


class ClusterSpec:
    """Base ClusterSpec used by Throne.

    Specs own two responsibilities:
    1) Produce the JobTrait via get_job()
    2) Configure monarch_hyperactor (transport, etc.) for both controller and
       workers.
    """

    def get_job(self) -> JobTrait:
        raise NotImplementedError

    def configure(self) -> None:
        """Configure monarch_hyperactor for this cluster."""
        cfg = self.configure_kwargs
        if cfg:
            configure(**cfg)

    def on_init(self) -> None:
        """Optional hook executed before job.apply()."""
        return None

    @property
    def controller_env(self) -> Dict[str, str]:
        """Environment variables to set on the controller/submit side."""
        return {}

    @property
    def worker_env(self) -> Dict[str, str]:
        """Environment variables to propagate to workers (e.g., via Slurm)."""
        return {}

    @property
    def configure_kwargs(self) -> Dict[str, Any]:
        """Keyword args forwarded to monarch_hyperactor.config.configure()."""
        return {"default_transport": ChannelTransport.TcpWithHostname}
