"""
Cluster abstraction for managing long-lived compute resources.

A Cluster wraps Monarch's Jobs API and provides:
- Named clusters (e.g., "slurm" instead of job handles)
- State persistence (save/load cluster state)
- Easy access to HostMeshes for resource allocation
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

from monarch.job import job_load, JobTrait

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Cluster:
    """A named cluster backed by a Monarch job.

    Example:
        # Create from job definition
        from monarch.job import LocalJob
        job = LocalJob(hosts=["pool"])
        job.apply()

        cluster = Cluster(job, name="local", state_dir="/path/to/clusters")
        cluster.save()

        # Later, load it
        cluster = Cluster.load("local", state_dir="/path/to/clusters")
        hosts = cluster.get_mesh("pool")
    """

    def __init__(
        self,
        job: JobTrait,
        name: str,
        state_dir: Optional[Path] = None,
        cluster_id: Optional[str] = None,
    ):
        self._job = job
        self._name = name
        self._cluster_id = cluster_id or str(uuid.uuid4())[:8]
        self._state_dir = (
            Path(state_dir) if state_dir else Path.home() / ".forge" / "clusters"
        )
        self._cluster_path = self._state_dir / name
        self._state = None  # Lazy load job state

    @property
    def name(self) -> str:
        """Cluster name."""
        return self._name

    @property
    def cluster_id(self) -> str:
        """Unique cluster ID."""
        return self._cluster_id

    def get_mesh(self, mesh_name: str = "pool"):
        """Get a HostMesh by name.

        Args:
            mesh_name: Name of the mesh (corresponds to meshes dict in job definition)

        Returns:
            HostMesh object for spawning processes
        """
        if self._state is None:
            self._state = self._job.state()
            self.print_job_info()

        return getattr(self._state, mesh_name)

    def print_job_info(self):
        """logger.info information about the underlying job."""
        try:
            # Get job type
            job_type = type(self._job).__name__

            # Get mesh information
            meshes = []
            for attr in dir(self._state):
                if not attr.startswith("_"):
                    mesh = getattr(self._state, attr, None)
                    if hasattr(mesh, "__class__") and "HostMesh" in str(mesh.__class__):
                        # Try to get host count
                        try:
                            # HostMesh should have hosts dimension
                            num_hosts = (
                                len(mesh._hosts) if hasattr(mesh, "_hosts") else "?"
                            )
                            meshes.append(f"{attr}({num_hosts} hosts)")
                        except:
                            meshes.append(attr)

            mesh_info = ", ".join(meshes) if meshes else "unknown"

            logger.info(f"Cluster '{self._name}' [{self._cluster_id}]:")
            logger.info(f"  Type: {job_type}")
            logger.info(f"  Meshes: {mesh_info}")

            # For SLURM jobs, try to get node names
            if job_type == "SlurmJob":
                logger.info(f"  Scheduler: SLURM")
                # TODO: Extract SLURM node names from job state
            elif job_type == "LocalJob":
                logger.info(f"  Scheduler: Local")

        except Exception as e:
            logger.debug(f"Could not extract job info: {e}")

    def save(self):
        """Save cluster state to disk for later loading.

        Saves to: {state_dir}/{name}/job_state.pkl

        Also saves cluster metadata (ID, creation time, etc).
        """
        self._cluster_path.mkdir(parents=True, exist_ok=True)
        state_path = self._cluster_path / "job_state.pkl"
        metadata_path = self._cluster_path / "metadata.txt"

        # Save job state
        self._job.dump(str(state_path))

        # Save metadata
        import datetime

        metadata = f"""Cluster: {self._name}
ID: {self._cluster_id}
Created: {datetime.datetime.now().isoformat()}
Job Type: {type(self._job).__name__}
State Path: {state_path}
"""
        metadata_path.write_text(metadata)

        logger.info(f"Cluster '{self._name}' saved")
        logger.info(f"ID: {self._cluster_id}")
        logger.info(f"Path: {self._cluster_path}")

    @classmethod
    def load(cls, name: str, state_dir: Optional[Path] = None) -> "Cluster":
        """Load a previously saved cluster.

        Args:
            name: Cluster name (e.g., "slurm", "local")
            state_dir: Directory where cluster states are stored (default: ~/.forge/clusters)

        Returns:
            Cluster instance

        Raises:
            FileNotFoundError: If cluster state file doesn't exist
            RuntimeError: If cluster state is invalid or corrupted
        """
        base_state_dir = (
            Path(state_dir) if state_dir else Path.home() / ".forge" / "clusters"
        )
        cluster_path = base_state_dir / name
        state_path = cluster_path / "job_state.pkl"
        metadata_path = cluster_path / "metadata.txt"

        if not state_path.exists():
            raise FileNotFoundError(
                f"Cluster '{name}' not found.\n"
                f"Expected state file at: {state_path}\n"
                f"Available clusters: {cls._list_clusters(base_state_dir)}"
            )

        # Load metadata if available
        cluster_id = None
        if metadata_path.exists():
            metadata = metadata_path.read_text()
            for line in metadata.split("\n"):
                if line.startswith("ID:"):
                    cluster_id = line.split(":", 1)[1].strip()

        # Load job state
        try:
            job = job_load(str(state_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load cluster '{name}' from {state_path}\n"
                f"Error: {e}\n"
                f"The cluster state may be corrupted."
            )

        logger.info(f"Loaded cluster '{name}'")
        if cluster_id:
            logger.info(f"ID: {cluster_id}")
        logger.info(f"Path: {cluster_path}")

        return cls(job, name, state_dir=base_state_dir, cluster_id=cluster_id)

    @classmethod
    def from_env(cls) -> "Cluster":
        """Load cluster from THRONE_CLUSTER environment variable.

        This is used by `throne run` to automatically provide cluster context.

        Returns:
            Cluster instance

        Raises:
            RuntimeError: If THRONE_CLUSTER env var is not set
        """
        import os

        cluster_name = os.environ.get("THRONE_CLUSTER")
        if not cluster_name:
            raise RuntimeError(
                "THRONE_CLUSTER environment variable not set.\n"
                "Use 'throne run <cluster> <script>' to run scripts with cluster context."
            )

        state_dir_str = os.environ.get("THRONE_STATE_DIR")
        state_dir = Path(state_dir_str) if state_dir_str else None

        return cls.load(cluster_name, state_dir=state_dir)

    @staticmethod
    def _list_clusters(state_dir: Path) -> str:
        """List available clusters in state directory."""
        if not state_dir.exists():
            return "none"

        clusters = []
        for d in state_dir.iterdir():
            if d.is_dir() and (d / "job_state.pkl").exists():
                clusters.append(d.name)

        return ", ".join(clusters) if clusters else "none"

    def kill(self):
        """Terminate the cluster job."""
        logger.info(f"Killing cluster '{self._name}'")
        self._job.kill()

    def info(self) -> dict:
        """Get cluster information.

        Returns:
            Dict with cluster metadata (meshes, status, etc)
        """
        # TODO: Extract useful info from job state
        # For now, just return basic info
        return {
            "name": self._name,
            "state_dir": str(self._state_dir),
        }

    @classmethod
    def from_file(
        cls, cluster_file: str, state_dir: Optional[Path] = None
    ) -> "Cluster":
        """Create cluster from a Python file defining a job.

        The cluster file should define a `job` variable (SlurmJob, LocalJob, etc).

        Example cluster file (slurm.py):
            from monarch.job import SlurmJob

            job = SlurmJob(
                meshes={"pool": 4},
                gpus_per_node=8,
                time_limit="24:00:00"
            )

        Args:
            cluster_file: Path to Python file defining the cluster
            state_dir: Directory where cluster states are stored (default: ~/.forge/clusters)

        Returns:
            Cluster instance (job not yet started)
        """
        import importlib.util

        cluster_path = Path(cluster_file)
        if not cluster_path.exists():
            raise FileNotFoundError(f"Cluster file not found: {cluster_file}")

        # Import the cluster definition
        spec = importlib.util.spec_from_file_location("cluster_def", cluster_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "job"):
            raise ValueError(
                f"Cluster file {cluster_file} must define a 'job' variable.\n"
                f"Example:\n"
                f"  from monarch.job import LocalJob\n"
                f"  job = LocalJob(hosts=['pool'])"
            )

        job = module.job
        cluster_name = cluster_path.stem  # "slurm.py" → "slurm"

        return cls(job, cluster_name, state_dir=state_dir)
