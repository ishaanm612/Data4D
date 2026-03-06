"""
Ground Detection Utilities for UnrealZoo

This module provides utilities to detect ground level by spawning test actors,
supporting both single-point and batch processing for speed.
"""

import numpy as np
import os
import json
from typing import Tuple, Optional, List, Dict


def detect_ground_height(
    client, x: float, y: float, scan_height: float = 2500.0, cam_id: int = 0
) -> Optional[float]:
    """
    Detect ground height at a specific (X, Y) coordinate by spawning a test actor.

    Args:
        client: UnrealCV client instance
        x: X coordinate to scan
        y: Y coordinate to scan
        scan_height: Height to spawn test actor (default: 2500)
        cam_id: Unused, kept for compatibility

    Returns:
        Ground Z coordinate, or None if detection failed
    """
    import time
    import random

    # Create unique probe name
    probe_name = f"probe_{random.randint(1000, 9999)}"

    try:
        # Spawn test actor
        client.new_obj("bp_character_C", probe_name, [x, y, scan_height], [0, 0, 0])

        # Wait for actors to fall and settle (increased for better stability)
        time.sleep(4.0)

        # Get landed position
        landed_loc = client.get_obj_location(probe_name)
        ground_z = landed_loc[2]

        # Clean up
        client.client.request(f"vset /object/{probe_name}/destroy")
        time.sleep(0.1)

        print(f"  Ground detected at Z={ground_z:.1f}")
        return ground_z

    except Exception as e:
        print(f"  Failed to detect ground at ({x:.1f}, {y:.1f}): {e}")
        # Try to clean up
        try:
            client.client.request(f"vset /object/{probe_name}/destroy")
        except:
            pass
        return None


def detect_ground_height_batch(
    client,
    points: List[Tuple[float, float]],
    scan_height: float = 2500.0,
    batch_size: int = 3,
) -> Dict[Tuple[float, float], Optional[float]]:
    """
    Detect ground height at multiple (X, Y) coordinates simultaneously for speed.

    Args:
        client: UnrealCV client instance
        points: List of (x, y) tuples to scan
        scan_height: Height to spawn test actors (default: 2500)
        batch_size: Max actors to spawn at once (default: 5, safe for stability)


    Returns:
        Dictionary mapping (x, y) -> ground_z
    """
    import time
    import random

    results = {}

    # Process in batches
    for batch_start in range(0, len(points), batch_size):
        batch_points = points[batch_start : batch_start + batch_size]
        probe_names = []

        print(
            f"\nBatch {batch_start//batch_size + 1}/{(len(points)-1)//batch_size + 1}: Spawning {len(batch_points)} actors..."
        )

        try:
            # Spawn all actors in batch
            for x, y in batch_points:
                probe_name = f"probe_{random.randint(1000, 9999)}"
                client.new_obj(
                    "bp_character_C", probe_name, [x, y, scan_height], [0, 0, 0]
                )
                probe_names.append((probe_name, x, y))

            # Wait once for all to settle (increased for better stability)
            time.sleep(6.0)

            # Query all positions
            for probe_name, x, y in probe_names:
                try:
                    time.sleep(0.01)
                    landed_loc = client.get_obj_location(probe_name)
                    ground_z = landed_loc[2]
                    results[(x, y)] = ground_z
                    print(f"  [{x:.0f}, {y:.0f}] -> Z={ground_z:.1f}")
                except Exception as e:
                    print(f"  [{x:.0f}, {y:.0f}] -> Failed: {e}")
                    results[(x, y)] = None

            # Clean up all actors
            for probe_name, _, _ in probe_names:
                try:
                    client.client.request(f"vset /object/{probe_name}/destroy")
                except:
                    pass
            # Cooldown period to let engine recover
            time.sleep(2.0)

        except Exception as e:
            print(f"  Batch error: {e}")
            # Try to clean up on error
            for probe_name, x, y in probe_names:
                try:
                    client.client.request(f"vset /object/{probe_name}/destroy")
                except:
                    pass
                if (x, y) not in results:
                    results[(x, y)] = None

    return results


def generate_safe_spawn_grid(
    client,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    grid_spacing: float = 500.0,
    spawn_offset: float = 150.0,
    output_file: str = "safe_spawns.json",
    resume: bool = True,
) -> List[Dict]:
    """
    Generate a grid of safe spawn coordinates by scanning ground heights.

    Supports resuming interrupted scans by loading existing partial results.

    Args:
        client: UnrealCV client instance
        x_range: (min_x, max_x) range to scan
        y_range: (min_y, max_y) range to scan
        grid_spacing: Distance between scan points
        spawn_offset: Height offset above ground for spawning
        output_file: Output JSON file path
        resume: If True, resume from existing partial results (default: True)

    Returns:
        List of safe spawn coordinate dictionaries
    """
    safe_spawns = []
    scanned_points = set()

    x_min, x_max = x_range
    y_min, y_max = y_range

    # Try to load existing results for resume
    if resume and os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                existing_data = json.load(f)

            # Verify compatible scan parameters
            old_params = existing_data.get("scan_params", {})
            params_match = (
                old_params.get("x_range") == list(x_range)
                and old_params.get("y_range") == list(y_range)
                and old_params.get("grid_spacing") == grid_spacing
                and old_params.get("spawn_offset") == spawn_offset
            )

            if params_match:
                safe_spawns = existing_data.get("spawn_points", [])
                # Track which points have been scanned
                for spawn in safe_spawns:
                    scanned_points.add((spawn["x"], spawn["y"]))

                print(
                    f"\n✓ Resuming from existing scan with {len(safe_spawns)} points already found"
                )
                print(f"  ({len(scanned_points)} grid locations already scanned)")
            else:
                print(f"\n⚠ Existing file has incompatible parameters - starting fresh")
                safe_spawns = []
                scanned_points = set()
        except Exception as e:
            print(f"\n⚠ Could not load existing results: {e} - starting fresh")
            safe_spawns = []
            scanned_points = set()

    # Generate grid points
    x_points = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_points = np.arange(y_min, y_max + grid_spacing, grid_spacing)

    total_points = len(x_points) * len(y_points)
    print(f"\nTotal grid points: {total_points}")
    print(f"X range: [{x_min}, {x_max}], Y range: [{y_min}, {y_max}]")
    print(f"Grid spacing: {grid_spacing}")

    # Collect points that haven't been scanned yet
    all_points = [(x, y) for x in x_points for y in y_points]
    remaining_points = [
        (x, y) for x, y in all_points if (float(x), float(y)) not in scanned_points
    ]

    if not remaining_points:
        print(f"\n✓ All points already scanned! No work to do.")
        return safe_spawns

    print(f"\nRemaining points to scan: {len(remaining_points)}/{total_points}")
    print(f"Using batch processing (3 actors at a time with cooldown for stability)...")

    # Process in batches with incremental saving
    batch_size = 5
    for batch_start in range(0, len(remaining_points), batch_size):
        batch_points = remaining_points[batch_start : batch_start + batch_size]

        print(
            f"\nBatch {batch_start//batch_size + 1}/{(len(remaining_points)-1)//batch_size + 1}: Scanning {len(batch_points)} points..."
        )

        # Scan this batch
        batch_results = detect_ground_height_batch(
            client, batch_points, batch_size=batch_size
        )

        # Process and save results incrementally
        batch_found = 0
        for (x, y), ground_z in batch_results.items():
            scanned_points.add((x, y))

            if ground_z is not None:
                spawn_location = {
                    "x": float(x),
                    "y": float(y),
                    "z": float(ground_z + spawn_offset),
                    "ground_z": float(ground_z),
                }
                safe_spawns.append(spawn_location)
                batch_found += 1

        # Save progress after each batch
        output_data = {
            "scan_params": {
                "x_range": list(x_range),
                "y_range": list(y_range),
                "grid_spacing": grid_spacing,
                "spawn_offset": spawn_offset,
            },
            "spawn_points": safe_spawns,
            "progress": {
                "total_points": total_points,
                "scanned_points": len(scanned_points),
                "valid_points": len(safe_spawns),
                "completion_pct": (len(scanned_points) / total_points) * 100,
            },
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        # Progress update
        completion = (len(scanned_points) / total_points) * 100
        print(f"  Found {batch_found} valid points in this batch")
        print(
            f"  Progress: {len(scanned_points)}/{total_points} ({completion:.1f}%) - {len(safe_spawns)} valid spawns total"
        )

    print(
        f"\n✓ Scan complete! Found {len(safe_spawns)}/{total_points} valid spawn points"
    )
    print(f"  Results saved to: {output_file}")

    return safe_spawns


def load_safe_spawns(filename: str = "safe_spawns.json") -> List[Dict]:
    """
    Load safe spawn coordinates from JSON file.

    Args:
        filename: Path to safe spawns JSON file

    Returns:
        List of spawn coordinate dictionaries
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Safe spawns file not found: {filename}")

    with open(filename, "r") as f:
        data = json.load(f)

    return data["spawn_points"]


def get_random_safe_spawn(filename: str = "safe_spawns.json") -> Dict:
    """
    Get a random safe spawn location from the database.

    Args:
        filename: Path to safe spawns JSON file

    Returns:
        Random spawn coordinate dictionary
    """
    spawns = load_safe_spawns(filename)
    return spawns[np.random.randint(0, len(spawns))]


if __name__ == "__main__":
    import gym
    import gym_unrealcv

    print("=== Ground Detection Utility ===")
    print("Initializing UnrealZoo environment...")

    env = gym.make("UnrealAgent-MiddleEast-ContinuousColor-v0")
    env.reset()
    client = env.unwrapped.unrealcv

    print("\nGenerating safe spawn grid...")
    safe_spawns = generate_safe_spawn_grid(
        client,
        x_range=(-3000, 3000),
        y_range=(-3000, 3000),
        grid_spacing=500,
        spawn_offset=150,
        output_file="safe_spawns.json",
    )

    print("\n=== Grid Generation Complete ===")
    print(f"Total safe spawns: {len(safe_spawns)}")

    if safe_spawns:
        print("\nExample spawn locations:")
        for i, spawn in enumerate(safe_spawns[:5]):
            print(
                f"  {i+1}. X={spawn['x']:.1f}, Y={spawn['y']:.1f}, Z={spawn['z']:.1f}"
            )

    env.close()
