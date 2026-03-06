"""
Demo script for the UnrealZoo 4D Procedural Generation Pipeline.

This script demonstrates how to:
1. Configure a scene with heterogeneous actors
2. Set up multiple cameras (exocentric and egocentric)
3. Define scripted events and background motion
4. Record multi-view data with all modalities

Output structure:
output_dir/
├── camera_metadata.json
├── pipeline_metadata.json
├── sequence_metadata.json
├── cam0/
│   ├── rgb/
│   ├── depth/
│   ├── mask/
│   ├── flow/
│   └── metadata/
├── cam1/
│   └── ...
└── scene_metadata/
"""

import gym
import gym_unrealcv
import numpy as np
from pathlib import Path
from data4d.core.pipeline import Pipeline
from data4d.utils.ground_detection import load_safe_spawns


def create_pipeline_config(env_id, output_dir, safe_spawns):
    """
    Create a configuration for the 4D generation pipeline.

    Args:
        env_id: UnrealZoo environment ID
        output_dir: Directory to save output data
        safe_spawns: List of safe spawn locations loaded from safe_spawns.json
    """
    # Randomly select spawn locations for each actor type
    num_humans = 5
    num_vehicles = 3
    num_props = 10
    total_actors = num_humans + num_vehicles + num_props

    # Ensure we have enough safe spawns
    if len(safe_spawns) < total_actors:
        raise ValueError(
            f"Not enough safe spawns ({len(safe_spawns)}) for {total_actors} actors"
        )

    # Randomly select spawn indices
    selected_indices = np.random.choice(
        len(safe_spawns), size=total_actors, replace=False
    )

    # Partition indices for different actor types
    human_spawns = [safe_spawns[i] for i in selected_indices[:num_humans]]
    vehicle_spawns = [
        safe_spawns[i] for i in selected_indices[num_humans : num_humans + num_vehicles]
    ]
    prop_spawns = [
        safe_spawns[i] for i in selected_indices[num_humans + num_vehicles :]
    ]

    config = {
        "output_dir": output_dir,
        "random_seed": 42,
        # Scene Configuration
        "scene": {
            "actors": [
                # Humans
                {
                    "type": "BP_Human",
                    "count": num_humans,
                    "safe_spawns": human_spawns,  # Use pre-computed safe locations
                    "animations": ["Walking", "Running", "Idle"],
                },
                # Vehicles
                {
                    "type": "BP_Vehicle",
                    "count": num_vehicles,
                    "safe_spawns": vehicle_spawns,
                },
                # Props
                {
                    "type": "BP_Prop",
                    "count": num_props,
                    "safe_spawns": prop_spawns,
                },
            ],
            "env_randomization": {
                "lighting": {
                    "enabled": True,
                    "sun_angle_range": [30, 150],  # degrees
                    "sun_brightness_range": [0.5, 1.5],
                },
                "weather": {"enabled": False},  # Not yet implemented
            },
        },
        # Timeline Configuration
        "timeline": {
            "duration": 100,  # frames
            "fps": 30,
            # Scripted events
            "scripted_events": [
                {
                    "frame": 20,
                    "type": "play_animation",
                    "actor_id": "human_0",
                    "animation": "Walking",
                    "duration": 30,
                },
                {
                    "frame": 50,
                    "type": "set_velocity",
                    "actor_id": "vehicle_0",
                    "velocity": [100, 0, 0],  # cm/s
                },
            ],
            # Background motion
            "background_motion": {
                "enabled": True,
                "type": "random_walk",
                "params": {"speed_range": [10, 50], "turn_probability": 0.1},  # cm/s
            },
        },
        # Camera Configuration
        "cameras": [
            # Exocentric camera 1 (bird's eye view)
            {
                "id": "cam0",
                "type": "exo",
                "position": [0, 0, 500],  # [x, y, z] in cm
                "rotation": [-90, 0, 0],  # [pitch, yaw, roll] in degrees
                "fov": 90,
                "resolution": (1920, 1080),
            },
            # Exocentric camera 2 (side view)
            {
                "id": "cam1",
                "type": "exo",
                "position": [1000, 0, 200],
                "rotation": [-20, 90, 0],
                "fov": 90,
                "resolution": (1920, 1080),
            },
            # Egocentric camera (attached to human_0)
            {
                "id": "cam_ego",
                "type": "ego",
                "position": [0, 0, 170],  # Relative to human head
                "rotation": [0, 0, 0],
                "fov": 110,
                "resolution": (1280, 720),
            },
        ],
    }

    return config


def main():
    # Configuration
    env_id = "UnrealAgent-MiddleEast-ContinuousColor-v0"
    output_dir = "./output/4d_demo"

    print("=" * 60)
    print("UnrealZoo 4D Procedural Generation Pipeline - Demo")
    print("=" * 60)

    # Initialize environment
    print("\n[1/4] Initializing UnrealZoo environment...")
    env = gym.make(env_id)
    env.reset()

    # Get UnrealCV client
    unreal_client = env.unwrapped.unrealcv
    print(f"✓ Connected to {env_id}")

    # Load safe spawn points
    print("\n[2/5] Loading safe spawn points...")
    safe_spawns = load_safe_spawns("safe_spawns.json")
    print(f"✓ Loaded {len(safe_spawns)} safe spawn locations")

    # Create pipeline configuration
    print("\n[3/5] Creating pipeline configuration...")
    config = create_pipeline_config(env_id, output_dir, safe_spawns)
    print(f"✓ Configuration created")
    print(
        f"  - Actors: {sum(actor['count'] for actor in config['scene']['actors'])} total"
    )
    print(f"  - Cameras: {len(config['cameras'])}")
    print(
        f"  - Duration: {config['timeline']['duration']} frames @ {config['timeline']['fps']} FPS"
    )

    # Initialize pipeline
    print("\n[4/5] Initializing pipeline...")
    pipeline = Pipeline(unreal_client, config)
    print("✓ Pipeline initialized")

    # Run pipeline
    print("\n[5/5] Running pipeline...")
    try:
        pipeline.run()
        print("\n✓ Pipeline completed successfully!")
        print(f"\nOutput saved to: {output_dir}")
        print("\nGenerated data structure:")
        print("  - RGB images")
        print("  - Depth maps")
        print("  - Segmentation masks")
        print("  - Optical flow (if v1.0.4+)")
        print("  - Camera poses & intrinsics")
        print("  - Scene metadata")

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up...")
        pipeline.cleanup()
        env.close()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    main()
