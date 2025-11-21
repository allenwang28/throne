"""Example: Running distributed PyTorch on a Throne cluster.

This demonstrates:
1. Loading a cluster
2. Getting a ProcMesh for distributed training
3. Spawning actors that perform collective operations
"""

import os
import torch
import torch.distributed as dist
from throne import Cluster
from monarch.actor import Actor, endpoint, context
import socket


class DistributedWorker(Actor):
    """Actor that performs distributed training operations."""

    @endpoint
    async def run_allreduce(
        self, world_size: int, master_addr: str, backend: str, per_host_workers: int
    ):
        """Perform an all-reduce operation.

        Args:
            world_size: Total number of processes
            master_addr: Address for the rendezvous (rank 0 host)
            backend: torch.distributed backend to use
            per_host_workers: Number of worker processes per host
        """
        # Get rank from Monarch context
        message_rank = context().message_rank
        # message_rank is a Point with a .rank attribute for flat index
        rank = message_rank.rank

        print(f"[Rank {rank}/{world_size}] Worker starting...")

        # Set up environment variables for torch.distributed
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = "29500"

        # Initialize process group
        dist.init_process_group(backend=backend)

        print(f"[Rank {rank}/{world_size}] Process group initialized")

        # Place tensors on the right device for the chosen backend
        local_rank = rank % per_host_workers
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        # Create a tensor with this rank's value
        tensor = torch.tensor([float(rank)], device=device)
        print(f"[Rank {rank}/{world_size}] Before all_reduce: {tensor.item()}")

        # Perform all-reduce (sum)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        print(f"[Rank {rank}/{world_size}] After all_reduce: {tensor.item()}")
        print(f"[Rank {rank}/{world_size}] Expected sum: {sum(range(world_size))}")

        # Cleanup
        dist.destroy_process_group()
        print(f"[Rank {rank}/{world_size}] Worker finished")

        return tensor.item()


async def main():
    """Main entry point - load cluster and spawn workers."""

    # Load the cluster from environment (set by `throne run`)
    # Or fall back to explicit name for direct execution
    import os
    if os.environ.get("THRONE_CLUSTER"):
        print("Loading cluster from environment...")
        cluster = Cluster.from_env()
    else:
        print("Loading cluster by name...")
        cluster = Cluster.load("local_cluster")

    # Get the host mesh
    print("Getting host mesh...")
    host_mesh = cluster.get_mesh("pool")
    print(f"Got host mesh {host_mesh}")
    # Ensure host connections are established before spawning further
    try:
        await host_mesh.initialized
    except Exception:
        pass

    # Infer hosts from the underlying SlurmJob if available
    master_addr = os.environ.get("MASTER_ADDR")
    job = getattr(cluster, "_job", None)
    job_hosts = getattr(job, "_all_hostnames", None)
    if job_hosts:
        num_hosts = len(job_hosts)
        if not master_addr:
            master_addr = job_hosts[0]
    else:
        # Fallback: best-effort from extent
        sizes = getattr(host_mesh, "extent", {}) or {}
        try:
            # extent is usually a dict-like mapping dimension -> size
            if hasattr(sizes, "values"):
                num_hosts = 1
                for v in sizes.values():
                    num_hosts *= int(v)
            else:
                num_hosts = 1
        except Exception:
            num_hosts = 1
    if not master_addr:
        # Final fallback: local host
        master_addr = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = master_addr

    gpus_per_host = torch.cuda.device_count()
    per_host_workers = gpus_per_host if gpus_per_host > 0 else 1
    world_size = num_hosts * per_host_workers

    backend = "nccl" if gpus_per_host > 0 else "gloo"

    print(f"Detected hosts={num_hosts}, gpus/host={gpus_per_host}, backend={backend}, master_addr={master_addr}")

    # Spawn processes on the mesh
    print(f"Spawning {world_size} workers (per_host={per_host_workers})...")
    proc_mesh = host_mesh.spawn_procs(per_host={"workers": per_host_workers})

    print(f"Workers spawned on mesh: {proc_mesh.extent}")

    # Spawn actors on the proc mesh
    actors = proc_mesh.spawn("workers", DistributedWorker)

    # Run the all-reduce on all workers
    print("Running all-reduce on all workers...")
    results = await actors.run_allreduce.call(
        world_size, master_addr, backend, per_host_workers
    )

    print(f"All-reduce results: {results}")
    print("All workers completed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
