"""Local cluster definition for testing."""

from monarch.job import LocalJob
from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from throne.spec import ClusterSpec


class LocalSpec(ClusterSpec):
    def __init__(self, job):
        self.job = job

    def get_job(self):
        return self.job

    @property
    def configure_kwargs(self):
        return {"default_transport": ChannelTransport.Local}


# Define a simple local job with one mesh
job = LocalJob(hosts=["pool"])
spec = LocalSpec(job)
