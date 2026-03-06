import logging
import json
import numpy as np
from pathlib import Path
from data4d.core.generator import SceneGenerator
from data4d.core.director import Director
from data4d.core.recorder import MultiViewRecorder


class Pipeline:
    """
    Main orchestrator for the 4D procedural generation pipeline.
    Coordinates SceneGenerator, Director, and MultiViewRecorder.
    """

    def __init__(self, unreal_client, config):
        """
        Args:
            unreal_client: UnrealCV API client
            config: Dict with pipeline configuration:
                {
                    'output_dir': '/path/to/output',
                    'scene': {
                        'actors': [...],  # Actor spawn configs
                        'env_randomization': {...}  # Environment settings
                    },
                    'timeline': {
                        'duration': 100,  # frames
                        'fps': 30,
                        'scripted_events': [...],
                        'background_motion': {...}
                    },
                    'cameras': [...],  # Camera configurations
                    'random_seed': 42
                }
        """
        self.unreal = unreal_client
        self.config = config

        # Set random seed for reproducibility
        if "random_seed" in config:
            np.random.seed(config["random_seed"])

        # Initialize components
        self.generator = SceneGenerator(unreal_client)

        self.director = Director(unreal_client)

        self.recorder = MultiViewRecorder(
            unreal_client, config["output_dir"], config["cameras"]
        )

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def generate_scene(self):
        """
        Step 1: Populate the scene with actors and randomize environment.
        """
        self.logger.info("=== Step 1: Generating Scene ===")

        # Spawn actors from configuration
        self.generator.spawn_from_config(self.config["scene"]["actors"])
        spawned_actors = self.generator.get_spawned_actors()
        self.logger.info(f"Spawned {len(spawned_actors)} actors")

        # Apply environment randomization
        env_rand = self.config["scene"].get("env_randomization", {})
        if env_rand.get("lighting", {}).get("enabled", False):
            self.generator.randomize_environment()

        return spawned_actors

    def setup_timeline(self, spawned_actors):
        """
        Step 2: Set up the timeline with scripted events and background motion.
        """
        self.logger.info("=== Step 2: Setting Up Timeline ===")

        timeline_config = self.config["timeline"]

        # Add scripted events
        if "scripted_events" in timeline_config:
            for event in timeline_config["scripted_events"]:
                self.director.add_event(event)
            self.logger.info(
                f"Added {len(timeline_config['scripted_events'])} scripted events"
            )

        # Add background motion for all spawned actors
        if timeline_config.get("background_motion", {}).get("enabled", False):
            motion_config = timeline_config["background_motion"]
            for actor_id in spawned_actors:
                # Random walk parameters
                self.director.add_background_motion(
                    actor_id,
                    motion_type=motion_config.get("type", "random_walk"),
                    params=motion_config.get("params", {}),
                )
            self.logger.info(
                f"Added background motion for {len(spawned_actors)} actors"
            )

    def setup_cameras(self):
        """
        Step 3: Position and configure all cameras.
        """
        self.logger.info("=== Step 3: Setting Up Cameras ===")
        self.recorder.setup_cameras()

    def run_recording(self):
        """
        Step 4: Execute the timeline and record all modalities.
        """
        self.logger.info("=== Step 4: Recording Sequence ===")

        timeline_config = self.config["timeline"]
        duration = timeline_config["duration"]
        fps = timeline_config["fps"]

        # Record sequence with synchronized timeline execution
        for frame_idx in range(duration):
            # Update timeline (trigger events, animate actors)
            self.director.update_timeline(frame_idx)

            # Capture all modalities from all cameras
            frame_meta = self.recorder.capture_frame()

            # Log progress
            if (frame_idx + 1) % 10 == 0:
                self.logger.info(f"Progress: {frame_idx + 1}/{duration} frames")

        self.logger.info(f"Recording complete: {duration} frames @ {fps} FPS")

    def save_pipeline_metadata(self):
        """
        Save complete pipeline configuration and metadata.
        """
        output_dir = Path(self.config["output_dir"])
        meta_path = output_dir / "pipeline_metadata.json"

        metadata = {
            "config": self.config,
            "spawned_actors": self.generator.spawned_actors,
            "timeline_events": [
                {
                    "frame": event["frame"],
                    "type": event["type"],
                    "actor_id": event.get("actor_id", "N/A"),
                }
                for event in self.director.events
            ],
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved pipeline metadata to {meta_path}")

    def run(self):
        """
        Execute the complete pipeline.
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting 4D Generation Pipeline")
        self.logger.info("=" * 60)

        # Step 1: Generate scene
        spawned_actors = self.generate_scene()

        # Step 2: Setup timeline
        self.setup_timeline(spawned_actors)

        # Step 3: Setup cameras
        self.setup_cameras()

        # Step 4: Run recording
        self.run_recording()

        # Save metadata
        self.save_pipeline_metadata()

        self.logger.info("=" * 60)
        self.logger.info("Pipeline Complete!")
        self.logger.info(f"Output saved to: {self.config['output_dir']}")
        self.logger.info("=" * 60)

    def cleanup(self):
        """
        Clean up resources and reset the environment.
        """
        self.logger.info("Cleaning up pipeline resources...")
        self.generator.reset()
        self.director.reset()
        self.recorder.reset()
