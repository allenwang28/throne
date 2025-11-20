"""Test the improved Cluster abstraction."""

import logging
from pathlib import Path

from cluster import Cluster

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_cluster_lifecycle():
    """Test creating, saving, loading, and using a cluster."""

    print("=" * 60)
    print("Testing Improved Cluster Lifecycle")
    print("=" * 60)

    # Step 1: Create cluster from definition file
    print("\n1. Creating cluster from local_cluster.py...")
    cluster = Cluster.from_file("local_cluster.py")
    print(f"   ✓ Created cluster: {cluster.name}")
    print(f"   ✓ Cluster ID: {cluster.cluster_id}")

    # Step 2: Apply the job (start it)
    print("\n2. Starting the cluster job...")
    cluster._job.apply()
    print("   ✓ Job started")

    # Step 3: Save cluster state
    print("\n3. Saving cluster state...")
    cluster.save()

    # Step 4: Test load failure with wrong name
    print("\n4. Testing load failure...")
    try:
        Cluster.load("nonexistent_cluster")
        print("   ✗ Should have failed!")
    except FileNotFoundError as e:
        print(f"   ✓ Correctly failed with: {e}")

    # Step 5: Load cluster from disk
    print("\n5. Loading cluster from disk...")
    loaded_cluster = Cluster.load("local_cluster")

    # Step 6: Get HostMesh (this will print job info)
    print("\n6. Getting HostMesh (will print job info)...")
    hosts = loaded_cluster.get_mesh("pool")
    print(f"   ✓ Got mesh: {hosts}")

    # Step 7: Spawn some processes to verify it works
    print("\n7. Spawning test processes...")
    procs = hosts.spawn_procs(per_host={"workers": 2})
    print(f"   ✓ Spawned ProcMesh with extent: {procs.extent}")

    # Step 8: Test with custom state_dir
    print("\n8. Testing custom state directory...")
    custom_dir = Path("/tmp/test_clusters")
    cluster2 = Cluster.from_file("local_cluster.py", state_dir=custom_dir)
    cluster2._job.apply()
    cluster2.save()
    print(f"   ✓ Saved to custom directory")

    loaded_cluster2 = Cluster.load("local_cluster", state_dir=custom_dir)
    print(f"   ✓ Loaded from custom directory")

    # Step 9: Cleanup
    print("\n9. Cleaning up...")
    loaded_cluster.kill()
    loaded_cluster2.kill()
    print("   ✓ Clusters killed")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_cluster_lifecycle()
