"""
Batch generation script for creating multiple 4D sequences with variations.

This script demonstrates:
- Large-scale dataset generation
- Reproducible splits (train/val/test)
- Distributed rendering setup (multi-instance)
- Variation across scenes, lighting, and actor configurations
"""

import gym
import gym_unrealcv
import numpy as np
from pathlib import Path
from data4d.core.pipeline import Pipeline
import json
import multiprocessing as mp
from typing import List, Dict


class BatchGenerator:
    """
    Orchestrates batch generation of 4D sequences.
    """

    def __init__(self, base_config, output_root):
        self.base_config = base_config
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def create_split_metadata(self, total_sequences, split_ratios):
        """
        Create train/val/test split assignments.

        Args:
            total_sequences: Total number of sequences to generate
            split_ratios: Dict like {'train': 0.7, 'val': 0.15, 'test': 0.15}

        Returns:
            Dict mapping sequence_id to split name
        """
        splits = {}
        indices = np.random.permutation(total_sequences)

        train_end = int(total_sequences * split_ratios["train"])
        val_end = train_end + int(total_sequences * split_ratios["val"])

        for i, idx in enumerate(indices):
            if i < train_end:
                splits[idx] = "train"
            elif i < val_end:
                splits[idx] = "val"
            else:
                splits[idx] = "test"

        return splits

    def create_variation_config(self, sequence_id, base_config):
        """
        Create a variation of the base config for diversity.
        """
        config = base_config.copy()

        # Vary random seed
        config["random_seed"] = sequence_id

        # Vary actor counts
        for actor in config["scene"]["actors"]:
            variation = np.random.randint(-2, 3)
            actor["count"] = max(1, actor["count"] + variation)

        # Vary lighting
        if config["scene"]["env_randomization"].get("lighting", {}).get("enabled"):
            lighting = config["scene"]["env_randomization"]["lighting"]
            lighting["sun_brightness"] = np.random.uniform(
                lighting["sun_brightness_range"][0], lighting["sun_brightness_range"][1]
            )

        # Update output directory
        config["output_dir"] = str(self.output_root / f"sequence_{sequence_id:04d}")

        return config

    def generate_sequence(self, args):
        """
        Generate a single sequence (for multiprocessing).
        """
        sequence_id, config, env_id = args

        print(f"[Sequence {sequence_id:04d}] Starting generation...")

        try:
            # Initialize environment
            env = gym.make(env_id)
            env.reset()
            unreal_client = env.unwrapped.unrealcv

            # Run pipeline
            pipeline = Pipeline(unreal_client, config)
            pipeline.run()

            # Cleanup
            pipeline.cleanup()
            env.close()

            print(f"[Sequence {sequence_id:04d}] ✓ Complete")
            return sequence_id, True, None

        except Exception as e:
            print(f"[Sequence {sequence_id:04d}] ✗ Failed: {e}")
            return sequence_id, False, str(e)

    def generate_batch(self, env_id, num_sequences, num_workers=1, split_ratios=None):
        """
        Generate multiple sequences in parallel.

        Args:
            env_id: UnrealZoo environment ID
            num_sequences: Number of sequences to generate
            num_workers: Number of parallel workers (set to 1 for sequential)
            split_ratios: Optional dict for train/val/test splits
        """
        if split_ratios is None:
            split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

        # Create splits
        splits = self.create_split_metadata(num_sequences, split_ratios)

        # Prepare configs
        configs = []
        for seq_id in range(num_sequences):
            config = self.create_variation_config(seq_id, self.base_config)
            config["split"] = splits[seq_id]
            configs.append((seq_id, config, env_id))

        # Generate sequences
        if num_workers > 1:
            print(f"Starting batch generation with {num_workers} workers...")
            with mp.Pool(num_workers) as pool:
                results = pool.map(self.generate_sequence, configs)
        else:
            print("Starting sequential generation...")
            results = [self.generate_sequence(cfg) for cfg in configs]

        # Save batch metadata
        batch_meta = {
            "env_id": env_id,
            "num_sequences": num_sequences,
            "splits": splits,
            "results": [
                {
                    "sequence_id": seq_id,
                    "success": success,
                    "error": error,
                    "split": splits[seq_id],
                }
                for seq_id, success, error in results
            ],
        }

        meta_path = self.output_root / "batch_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(batch_meta, f, indent=2)

        print(f"\nBatch generation complete!")
        print(f"Successful: {sum(1 for _, s, _ in results if s)}/{num_sequences}")
        print(f"Metadata saved to: {meta_path}")


def main():
    # Base configuration (will be varied for each sequence)
    base_config = {
        "scene": {
            "actors": [
                {
                    "type": "BP_Human",
                    "count": 5,
                    "spawn_area": {"x": [-500, 500], "y": [-500, 500], "z": [0, 0]},
                },
                {
                    "type": "BP_Vehicle",
                    "count": 2,
                    "spawn_area": {"x": [-1000, 1000], "y": [-1000, 1000], "z": [0, 0]},
                },
                {
                    "type": "BP_Prop",
                    "count": 8,
                    "spawn_area": {"x": [-800, 800], "y": [-800, 800], "z": [0, 100]},
                },
            ],
            "env_randomization": {
                "lighting": {
                    "enabled": True,
                    "sun_angle_range": [30, 150],
                    "sun_brightness_range": [0.5, 1.5],
                }
            },
        },
        "timeline": {
            "duration": 60,
            "fps": 30,
            "scripted_events": [],
            "background_motion": {
                "enabled": True,
                "type": "random_walk",
                "params": {"speed_range": [10, 50], "turn_probability": 0.1},
            },
        },
        "cameras": [
            {
                "id": "cam0",
                "type": "exo",
                "position": [0, 0, 500],
                "rotation": [-90, 0, 0],
                "fov": 90,
                "resolution": (1920, 1080),
            },
            {
                "id": "cam1",
                "type": "exo",
                "position": [1000, 0, 200],
                "rotation": [-20, 90, 0],
                "fov": 90,
                "resolution": (1920, 1080),
            },
        ],
    }

    # Batch generation settings
    env_id = "UnrealAgent-MiddleEast-ContinuousColor-v0"
    output_root = "./output/batch_4d"
    num_sequences = 10
    num_workers = 1  # Increase for parallel generation (requires multiple UE instances)

    print("=" * 60)
    print("UnrealZoo 4D Batch Generation")
    print("=" * 60)
    print(f"Environment: {env_id}")
    print(f"Sequences: {num_sequences}")
    print(f"Workers: {num_workers}")
    print(f"Output: {output_root}")
    print("=" * 60)

    # Run batch generation
    generator = BatchGenerator(base_config, output_root)
    generator.generate_batch(env_id, num_sequences, num_workers)


if __name__ == "__main__":
    main()
