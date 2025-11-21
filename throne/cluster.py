"""
Cluster abstraction for managing long-lived compute resources.

A Cluster wraps Monarch's Jobs API and provides:
- Named clusters (e.g., "slurm" instead of job handles)
- State persistence (save/load cluster state)
- Easy access to HostMeshes for resource allocation
"""

import datetime
import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple

from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from monarch._rust_bindings.monarch_hyperactor.config import configure
from monarch.job import JobTrait, job_load
from throne.spec import ClusterSpec

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
            # If the job is inactive, clean up metadata and surface the error
            if hasattr(self._job, "active") and not self._job.active:
                self.cleanup()
                raise RuntimeError(
                    f"Underlying job for cluster '{self._name}' is not active. "
                    "Cluster metadata has been removed."
                )

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

    def _render_sbatch(self) -> str:
        """Return the sbatch-style directives if available (for SlurmJob)."""
        job_type = type(self._job).__name__
        if job_type != "SlurmJob":
            return ""

        lines = []
        append = lines.append
        append(f"#SBATCH --job-name={getattr(self._job, '_job_name', '?')}")
        append(f"#SBATCH --ntasks-per-node={getattr(self._job, '_ntasks_per_node', '?')}")

        nodes = sum(getattr(self._job, "_meshes", {}).values()) if hasattr(self._job, "_meshes") else "?"
        append(f"#SBATCH --nodes={nodes}")

        log_dir = getattr(self._job, "_log_dir", None)
        if log_dir:
            append(f"#SBATCH --output={Path(log_dir)}/slurm_%j.out")
            append(f"#SBATCH --error={Path(log_dir)}/slurm_%j.err")

        time_limit = getattr(self._job, "_time_limit", None)
        if time_limit:
            append(f"#SBATCH --time={time_limit}")

        partition = getattr(self._job, "_partition", None)
        if partition:
            append(f"#SBATCH --partition={partition}")

        gpus_per_node = getattr(self._job, "_gpus_per_node", None)
        if gpus_per_node is not None:
            append(f"#SBATCH --gpus-per-node={gpus_per_node}")

        cpus_per_task = getattr(self._job, "_cpus_per_task", None)
        if cpus_per_task is not None:
            append(f"#SBATCH --cpus-per-task={cpus_per_task}")

        mem = getattr(self._job, "_mem", None)
        if mem is not None:
            append(f"#SBATCH --mem={mem}")

        exclusive = getattr(self._job, "_exclusive", None)
        if exclusive:
            append("#SBATCH --exclusive")

        slurm_args = getattr(self._job, "_slurm_args", None)
        if slurm_args:
            for arg in slurm_args:
                append(f"#SBATCH {arg}")

        return "\n".join(lines)

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

        metadata = self._build_metadata(state_path)
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
        metadata = {}
        if metadata_path.exists():
            metadata = cls._parse_metadata(metadata_path)
            cluster_id = metadata.get("ID")
            cluster_file = metadata.get("Cluster File")
        else:
            cluster_file = None

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

        cluster = cls(job, name, state_dir=base_state_dir, cluster_id=cluster_id)
        spec_obj = None
        if metadata:
            if cluster_file:
                cluster._cluster_file = Path(cluster_file)
            if "Controller Env" in metadata:
                try:
                    cluster._controller_env = eval(metadata["Controller Env"])
                except Exception:
                    cluster._controller_env = {}
            if "Worker Env" in metadata:
                try:
                    cluster._worker_env = eval(metadata["Worker Env"])
                except Exception:
                    cluster._worker_env = {}

        # Rehydrate spec/on_init if cluster file is known
        if cluster_file:
            try:
                spec_obj = cls._load_spec_from_path(Path(cluster_file))
                controller_env, worker_env, job_from_spec = cls._prepare_spec(spec_obj)
                cls._apply_worker_env(job, worker_env)
                cluster._spec = spec_obj
                cluster._controller_env = controller_env or getattr(
                    cluster, "_controller_env", {}
                )
                cluster._worker_env = worker_env or getattr(
                    cluster, "_worker_env", {}
                )
            except Exception as e:
                logger.warning(f"Failed to reload spec from {cluster_file}: {e}")

        if spec_obj is None:
            # Fallback for legacy state without spec metadata or failed imports
            cls._configure_from_job(job)

        return cluster

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

    def _job_metadata(self) -> dict:
        """Return job-specific metadata for persistence."""
        job_type = type(self._job).__name__
        metadata = {}

        if job_type == "SlurmJob":
            meshes = getattr(self._job, "_meshes", None)
            if meshes:
                metadata["Meshes"] = ", ".join(f"{k}={v}" for k, v in meshes.items())
                metadata["Nodes"] = sum(meshes.values())

            metadata["Partition"] = getattr(self._job, "_partition", None)
            metadata["Time Limit"] = getattr(self._job, "_time_limit", None)
            metadata["GPUs per node"] = getattr(self._job, "_gpus_per_node", None)
            metadata["Tasks per node"] = getattr(self._job, "_ntasks_per_node", None)
            metadata["Python"] = getattr(self._job, "_python_exe", None)
            metadata["Job Name"] = getattr(self._job, "_job_name", None)
            metadata["Port"] = getattr(self._job, "_port", None)
            metadata["Log Dir"] = getattr(self._job, "_log_dir", None)
            metadata["Exclusive"] = getattr(self._job, "_exclusive", None)
            metadata["CPUs per task"] = getattr(self._job, "_cpus_per_task", None)
            metadata["Memory"] = getattr(self._job, "_mem", None)

            job_id = getattr(self._job, "_slurm_job_id", None)
            if job_id:
                metadata["Job ID"] = job_id

            slurm_args = getattr(self._job, "_slurm_args", None)
            if slurm_args:
                metadata["SLURM Args"] = " ".join(slurm_args)

        metadata["Job Type"] = job_type
        return {k: v for k, v in metadata.items() if v is not None}

    def _build_metadata(self, state_path: Path) -> str:
        """Format metadata as key-value lines."""
        metadata = {
            "Cluster": self._name,
            "ID": self._cluster_id,
            "Created": datetime.datetime.now().isoformat(),
            "State Path": state_path,
        }
        if hasattr(self, "_cluster_file"):
            metadata["Cluster File"] = str(getattr(self, "_cluster_file"))
        if getattr(self, "_controller_env", None):
            metadata["Controller Env"] = repr(dict(self._controller_env))
        if getattr(self, "_worker_env", None):
            metadata["Worker Env"] = repr(dict(self._worker_env))
        metadata.update(self._job_metadata())
        sbatch = self._render_sbatch()
        if sbatch:
            metadata["SBATCH"] = sbatch.replace("\n", " | ")
        lines = [f"{key}: {value}" for key, value in metadata.items()]
        return "\n".join(lines) + "\n"

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

    def cleanup(self):
        """Delete cluster metadata/state directory."""
        if self._cluster_path.exists():
            import shutil

            try:
                shutil.rmtree(self._cluster_path)
                logger.info(f"Removed cluster state at {self._cluster_path}")
            except Exception as e:
                logger.warning(f"Failed to remove cluster state at {self._cluster_path}: {e}")

    @staticmethod
    def _parse_metadata(metadata_path: Path) -> dict:
        """Read key:value metadata pairs from file."""
        metadata = {}
        if not metadata_path.exists():
            return metadata

        content = metadata_path.read_text()
        for line in content.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        return metadata

    @classmethod
    def from_file(
        cls, cluster_file: str, state_dir: Optional[Path] = None
    ) -> "Cluster":
        """Create cluster from a Python file defining a job.

        The cluster file should define a `spec: ClusterSpec` that can configure
        monarch and return a JobTrait via get_job().

        Example cluster file (slurm.py):
            from monarch.job import SlurmJob

            class MySpec(ClusterSpec):
                def configure(self):
                    configure(default_transport=ChannelTransport.TcpWithHostname)

                def get_job(self):
                    return SlurmJob(meshes={"pool": 4})

            spec = MySpec()
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

        spec_obj = cls._load_spec_from_path(cluster_path)
        controller_env, worker_env, job = cls._prepare_spec(spec_obj)
        cls._apply_worker_env(job, worker_env)
        cluster_name = cluster_path.stem  # "slurm.py" → "slurm"

        cluster = cls(job, cluster_name, state_dir=state_dir)
        cluster._cluster_file = cluster_path
        cluster._spec = spec_obj
        cluster._controller_env = controller_env
        cluster._worker_env = worker_env

        return cluster

    @staticmethod
    def _load_spec_from_path(cluster_file: Path) -> ClusterSpec:
        """Load a ClusterSpec from a cluster definition file."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("cluster_def", cluster_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "spec"):
            raise ValueError(
                f"Cluster file {cluster_file} must define a 'spec' instance of ClusterSpec "
                "that configures monarch and returns a job via get_job()."
            )

        spec_obj = getattr(module, "spec")
        if not isinstance(spec_obj, ClusterSpec):
            raise TypeError(
                f"'spec' defined in {cluster_file} must be a ClusterSpec (got {type(spec_obj)})"
            )
        return spec_obj

    @staticmethod
    def _prepare_spec(spec_obj: ClusterSpec) -> Tuple[dict, dict, JobTrait]:
        """Run spec.configure/on_init and return controller/worker env and job."""
        spec_obj.configure()
        spec_obj.on_init()
        controller_env = getattr(spec_obj, "controller_env", {}) or {}
        worker_env = getattr(spec_obj, "worker_env", {}) or {}
        job = spec_obj.get_job()
        return controller_env, worker_env, job

    @staticmethod
    def _apply_worker_env(job: JobTrait, worker_env: dict) -> None:
        """
        Propagate worker env to schedulers that support it.

        For SlurmJob we inject an --export line so sbatch propagates the env to
        worker tasks. No-op if env is empty or already exported.
        """
        if not worker_env:
            return
        try:
            job_type = type(job).__name__
            if job_type != "SlurmJob":
                return
            export_pairs = [f"{k}={v}" for k, v in worker_env.items()]
            export_arg = f"--export=ALL,{','.join(export_pairs)}"
            current_args = list(getattr(job, "_slurm_args", ()))
            if any(arg.startswith("--export") for arg in current_args):
                return
            current_args.append(export_arg)
            job._slurm_args = tuple(current_args)  # type: ignore[attr-defined]
        except Exception as e:
            logger.debug(f"Failed to apply worker env to job: {e}")

    @staticmethod
    def _configure_from_job(job: JobTrait) -> None:
        """Best-effort configure when no spec is available (e.g., legacy saves)."""
        try:
            job_type = type(job).__name__
            if job_type == "SlurmJob":
                transport = ChannelTransport.TcpWithHostname
            elif job_type == "LocalJob":
                transport = ChannelTransport.Local
            else:
                return
            configure(default_transport=transport)
        except Exception as e:
            logger.debug(f"Failed to configure from job type: {e}")
