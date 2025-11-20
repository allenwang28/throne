"""Local cluster definition for testing."""

from monarch.job import LocalJob

# Define a simple local job with one mesh
job = LocalJob(hosts=["pool"])
