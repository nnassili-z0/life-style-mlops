#!/usr/bin/env python3
"""
One-time cleanup script for existing failed DAG artifacts.
This script archives artifacts from previously failed DAG runs.
"""

import os
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path

ARTIFACT_DIR = "/opt/airflow/demo_artifacts"

def cleanup_existing_failed_runs():
    """Archive artifacts from existing failed DAG runs."""
    print("Starting cleanup of existing failed DAG runs...")

    artifact_base = Path(ARTIFACT_DIR)
    if not artifact_base.exists():
        print(f"Artifact directory {ARTIFACT_DIR} does not exist")
        return

    archive_base = artifact_base / "archive"
    failed_archive = archive_base / "failed"
    failed_archive.mkdir(parents=True, exist_ok=True)

    # Get all timestamped directories (potential DAG runs)
    dag_dirs = []
    for item in artifact_base.iterdir():
        if item.is_dir() and not item.name.startswith("archive"):
            try:
                # Try to parse as timestamp directory
                datetime.strptime(item.name, "%Y%m%d_%H%M%S")
                dag_dirs.append(item)
            except ValueError:
                continue

    print(f"Found {len(dag_dirs)} potential DAG run directories")

    archived_count = 0
    for dag_dir in sorted(dag_dirs):
        ts = dag_dir.name

        # Check if this looks like a failed run by examining the directory structure
        # Failed runs typically won't have all the final tasks completed
        has_register = (dag_dir / "register").exists()
        has_summary = (dag_dir / "summary").exists()

        # If it doesn't have the final tasks, consider it failed
        if not (has_register and has_summary):
            print(f"Archiving failed run: {ts}")

            archive_path = failed_archive / f"failed_{ts}"

            try:
                shutil.move(str(dag_dir), str(archive_path))

                # Create failure summary
                failed_summary = {
                    "timestamp": ts,
                    "archive_location": str(archive_path),
                    "reason": "Archived during cleanup - missing final tasks",
                    "archived_at": datetime.now().isoformat()
                }

                summary_file = archive_path / "failure_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(failed_summary, f, indent=2)

                archived_count += 1
                print(f"  -> Moved to: {archive_path}")

            except Exception as e:
                print(f"  -> Error archiving {ts}: {e}")
        else:
            print(f"Keeping successful run: {ts}")

    print(f"Archived {archived_count} failed runs")

    # Clean up old archives (older than 30 days)
    print("Cleaning up old archives...")
    retention_days = 30
    cutoff_date = datetime.now() - timedelta(days=retention_days)

    removed_count = 0
    if failed_archive.exists():
        for item in failed_archive.iterdir():
            if item.is_dir() and item.name.startswith("failed_"):
                try:
                    timestamp_str = item.name.replace("failed_", "")
                    item_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                    if item_date < cutoff_date:
                        shutil.rmtree(item)
                        print(f"Removed old archive: {item.name}")
                        removed_count += 1
                except (ValueError, OSError) as e:
                    print(f"Could not process archive {item.name}: {e}")

    print(f"Removed {removed_count} old archives")
    print("Cleanup completed!")

if __name__ == "__main__":
    cleanup_existing_failed_runs()