"""
Generate Safe Spawn Coordinates

This script scans the UnrealZoo environment and generates a database of
safe spawn coordinates that can be used for actor placement.

Usage:
    python generate_safe_spawns.py
"""

import gym
import gym_unrealcv
import argparse
from unrealzoo.utils.ground_detection import generate_safe_spawn_grid


def main():
    parser = argparse.ArgumentParser(
        description="Generate safe spawn coordinates for UnrealZoo"
    )
    parser.add_argument(
        "--x-min", type=float, default=-5000, help="Minimum X coordinate"
    )
    parser.add_argument(
        "--x-max", type=float, default=5000, help="Maximum X coordinate"
    )
    parser.add_argument(
        "--y-min", type=float, default=-5000, help="Minimum Y coordinate"
    )
    parser.add_argument(
        "--y-max", type=float, default=5000, help="Maximum Y coordinate"
    )
    parser.add_argument("--spacing", type=float, default=250, help="Grid spacing")
    parser.add_argument(
        "--offset", type=float, default=150, help="Spawn height offset above ground"
    )
    parser.add_argument(
        "--output", type=str, default="safe_spawns.json", help="Output file path"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="UnrealAgent-Greek_Island-ContinuousColor-v0",
        help="Environment ID",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing partial results (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Start fresh, ignore existing results",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("UnrealZoo Safe Spawn Generator")
    print("=" * 60)
    print(f"\nEnvironment: {args.env}")
    print(f"Scan area: X[{args.x_min}, {args.x_max}], Y[{args.y_min}, {args.y_max}]")
    print(f"Grid spacing: {args.spacing}")
    print(f"Spawn offset: {args.offset}")
    print(f"Resume mode: {'Enabled' if args.resume else 'Disabled (starting fresh)'}\n")

    # Initialize environment
    print("Initializing environment...")
    env = gym.make(args.env)
    env.reset()
    client = env.unwrapped.unrealcv
    print("Environment ready!\n")

    # Generate safe spawn grid
    safe_spawns = generate_safe_spawn_grid(
        client,
        x_range=(args.x_min, args.x_max),
        y_range=(args.y_min, args.y_max),
        grid_spacing=args.spacing,
        spawn_offset=args.offset,
        output_file=args.output,
        resume=args.resume,
    )

    # Summary
    total_scanned = ((args.x_max - args.x_min) / args.spacing + 1) * (
        (args.y_max - args.y_min) / args.spacing + 1
    )
    success_rate = (len(safe_spawns) / total_scanned) * 100 if total_scanned > 0 else 0

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Total points scanned: {int(total_scanned)}")
    print(f"Valid spawn points: {len(safe_spawns)}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Output file: {args.output}")
    print("\nYou can now use these coordinates in your pipeline!")

    env.close()


if __name__ == "__main__":
    main()
