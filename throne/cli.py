#!/usr/bin/env python3
"""
Throne CLI - Cluster management tool.

Commands:
    throne start <cluster_file>  # Start a cluster
    throne stop <name>           # Stop a cluster
    throne list                  # List all clusters
    throne info <name>           # Show cluster info

Environment Variables:
    THRONE_STATE_DIR - Directory where cluster states are stored
                      (default: ~/.forge/clusters)
"""

import argparse
import os
import sys
from pathlib import Path

from .cluster import Cluster


def get_state_dir() -> Path:
    """Get state directory from env var or default."""
    state_dir = os.environ.get('THRONE_STATE_DIR')
    if state_dir:
        return Path(state_dir)
    return Path.home() / ".forge" / "clusters"


def cmd_start(args):
    """Start a cluster from a definition file."""
    try:
        # Load cluster definition
        state_dir = get_state_dir()
        cluster = Cluster.from_file(args.cluster_file, state_dir=state_dir)

        print(f"Starting cluster '{cluster.name}'...")

        # Apply the job (start it)
        cluster._job.apply()

        # Save state (this will print save info from Cluster.save())
        cluster.save()

        print(f"\n✓ Cluster '{cluster.name}' started successfully")
        print(f"  ID: {cluster.cluster_id}")
        print(f"  State saved to: {cluster._cluster_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error starting cluster: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_stop(args):
    """Stop a running cluster."""
    try:
        # Strip .py extension if provided (common mistake)
        cluster_name = args.name
        if cluster_name.endswith('.py'):
            cluster_name = cluster_name[:-3]
            print(f"Note: Using cluster name '{cluster_name}' (stripped .py extension)")

        # Load cluster
        state_dir = get_state_dir()
        cluster = Cluster.load(cluster_name, state_dir=state_dir)

        print(f"Stopping cluster '{cluster.name}'...")

        # Kill the job
        cluster.kill()

        print(f"✓ Cluster '{cluster.name}' stopped")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"\nHint: Use the cluster name, not the filename.", file=sys.stderr)
        print(f"      Example: throne stop {args.name.replace('.py', '')}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error stopping cluster: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list(args):
    """List all clusters."""
    state_dir = get_state_dir()

    if not state_dir.exists():
        print("No clusters found")
        return

    clusters = []
    for cluster_dir in state_dir.iterdir():
        if cluster_dir.is_dir() and (cluster_dir / "job_state.pkl").exists():
            metadata_path = cluster_dir / "metadata.txt"

            # Read metadata if available
            cluster_id = "unknown"
            created = "unknown"
            job_type = "unknown"

            if metadata_path.exists():
                metadata = metadata_path.read_text()
                for line in metadata.split('\n'):
                    if line.startswith('ID:'):
                        cluster_id = line.split(':', 1)[1].strip()
                    elif line.startswith('Created:'):
                        created = line.split(':', 1)[1].strip()
                    elif line.startswith('Job Type:'):
                        job_type = line.split(':', 1)[1].strip()

            clusters.append({
                'name': cluster_dir.name,
                'id': cluster_id,
                'created': created,
                'type': job_type,
            })

    if not clusters:
        print("No clusters found")
        return

    print(f"Clusters in {state_dir}:")
    print("-" * 80)
    print(f"{'NAME':<20} {'ID':<12} {'TYPE':<15} {'CREATED':<30}")
    print("-" * 80)

    for cluster in clusters:
        # Truncate created timestamp for display
        created = cluster['created'][:19] if len(cluster['created']) > 19 else cluster['created']
        print(f"{cluster['name']:<20} {cluster['id']:<12} {cluster['type']:<15} {created:<30}")


def cmd_info(args):
    """Show detailed cluster information."""
    try:
        # Strip .py extension if provided (common mistake)
        cluster_name = args.name
        if cluster_name.endswith('.py'):
            cluster_name = cluster_name[:-3]
            print(f"Note: Using cluster name '{cluster_name}' (stripped .py extension)")

        # Load cluster
        state_dir = get_state_dir()
        cluster = Cluster.load(cluster_name, state_dir=state_dir)

        print(f"\nCluster: {cluster.name}")
        print(f"ID: {cluster.cluster_id}")
        print(f"Path: {cluster._cluster_path}")

        # Try to show metadata
        metadata_path = cluster._cluster_path / "metadata.txt"
        if metadata_path.exists():
            print(f"\nMetadata:")
            metadata = metadata_path.read_text()
            for line in metadata.split('\n'):
                if line.strip() and not line.startswith('State Path'):
                    print(f"  {line}")

        # Note: We skip job state info because calling job.state()
        # spawns background threads that cause threading crashes on cleanup.
        # The metadata above contains the key info (type, created, etc.)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error getting cluster info: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_run(args):
    """Run a Python script with cluster context."""
    try:
        # Load cluster
        state_dir = get_state_dir()
        cluster = Cluster.load(args.cluster, state_dir=state_dir)

        print(f"Running {args.script} on cluster '{cluster.name}' [{cluster.cluster_id}]...")

        # Set environment variable so the script can find the cluster
        os.environ["THRONE_CLUSTER"] = cluster.name
        os.environ["THRONE_STATE_DIR"] = str(state_dir)

        # Execute the script
        import subprocess
        result = subprocess.run(
            [sys.executable, args.script] + args.script_args,
            env=os.environ.copy()
        )
        sys.exit(result.returncode)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running script: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Throne - Cluster management CLI\n\nEnvironment Variables:\n  THRONE_STATE_DIR - Directory for cluster states (default: ~/.forge/clusters)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # throne start
    start_parser = subparsers.add_parser('start', help='Start a cluster')
    start_parser.add_argument('cluster_file', help='Path to cluster definition file')
    start_parser.set_defaults(func=cmd_start)

    # throne stop
    stop_parser = subparsers.add_parser('stop', help='Stop a cluster')
    stop_parser.add_argument('name', help='Cluster name')
    stop_parser.set_defaults(func=cmd_stop)

    # throne list
    list_parser = subparsers.add_parser('list', help='List all clusters')
    list_parser.set_defaults(func=cmd_list)

    # throne info
    info_parser = subparsers.add_parser('info', help='Show cluster information')
    info_parser.add_argument('name', help='Cluster name')
    info_parser.set_defaults(func=cmd_info)

    # throne run
    run_parser = subparsers.add_parser('run', help='Run a script on a cluster')
    run_parser.add_argument('cluster', help='Cluster name')
    run_parser.add_argument('script', help='Python script to run')
    run_parser.add_argument('script_args', nargs='*', help='Arguments to pass to the script')
    run_parser.set_defaults(func=cmd_run)

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run command
    args.func(args)


if __name__ == '__main__':
    main()
