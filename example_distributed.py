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


class DistributedWorker(Actor):
    """Actor that performs distributed training operations."""

    @endpoint
    async def run_allreduce(self, world_size: int):
        """Perform an all-reduce operation.

        Args:
            world_size: Total number of processes
        """
        # Get rank from Monarch context
        message_rank = context().message_rank
        # message_rank is a Point with a .rank attribute for flat index
        rank = message_rank.rank

        print(f"[Rank {rank}/{world_size}] Worker starting...")

        # Set up environment variables for torch.distributed
        # We need to manually set these since Monarch doesn't do it automatically
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"  # For LocalJob
        os.environ["MASTER_PORT"] = "29500"

        # Initialize process group
        dist.init_process_group(backend="gloo")  # Use "nccl" for GPU

        print(f"[Rank {rank}/{world_size}] Process group initialized")

        # Create a tensor with this rank's value
        tensor = torch.tensor([float(rank)])
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

    # Spawn processes on the mesh
    # For LocalJob with 1 host, this will spawn multiple processes on localhost
    world_size = 4  # Number of processes to spawn

    print(f"Spawning {world_size} workers...")
    proc_mesh = host_mesh.spawn_procs(
        per_host={"workers": world_size}
    )

    print(f"Workers spawned on mesh: {proc_mesh.extent}")

    # Spawn actors on the proc mesh
    actors = proc_mesh.spawn("workers", DistributedWorker)

    # Run the all-reduce on all workers
    print("Running all-reduce on all workers...")
    results = await actors.run_allreduce.call(world_size)

    print(f"All-reduce results: {results}")
    print("All workers completed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
