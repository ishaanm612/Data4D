import logging
import time
import numpy as np


class SceneGenerator:
    """
    Responsible for populating the UnrealZoo scene with heterogeneous actors
    (humans, vehicles, props) and applying environment randomization.
    """

    def __init__(self, unreal_client):
        """
        Args:
            unreal_client: Instance of gym_unrealcv.envs.agent.character.Character_API
                           or compatible interface.
        """
        self.client = unreal_client
        self.spawned_actors = []
        self.logger = logging.getLogger(__name__)

    def reset(self):
        """
        Clears all actors spawned by this generator.
        """
        self.clean_scene()

    def clean_scene(self):
        """
        Destroys all actors tracked by this generator.
        """
        if not self.spawned_actors:
            return

        self.logger.info(f"Cleaning {len(self.spawned_actors)} actors...")
        # In gym-unrealcv, we might not have a direct 'destroy_all' for custom spawns
        # so we iterate.
        for actor_name in self.spawned_actors:
            # Check if destroy_obj method exists, otherwise use command
            if hasattr(self.client, "destroy_obj"):
                self.client.destroy_obj(actor_name)
            else:
                self.client.client.request(f"vset /object/{actor_name}/destroy")

        self.spawned_actors = []
        time.sleep(0.5)  # Allow engine to process destruction

    def spawn_from_config(self, config):
        """
        Spawns multiple actors based on a configuration dictionary.

        Args:
            config (list of dict): List of actor configurations.
                Example with spawn_area:
                [
                    {
                        "class_name": "bp_character_C",
                        "name": "walker_1",
                        "location": [0, 0, 200],
                        "rotation": [0, 0, 0],
                        "scale": [1, 1, 1], # optional
                        "physics": True     # optional
                    },
                    ...
                ]

                Example with safe_spawns (recommended):
                [
                    {
                        "type": "BP_Human",
                        "count": 5,
                        "safe_spawns": [
                            {"x": 100, "y": 200, "z": 150, "ground_z": 0},
                            ...
                        ],
                        "animations": ["Walking"]
                    },
                    ...
                ]
        """
        for actor_conf in config:
            # Check if using safe_spawns (new format)
            if "safe_spawns" in actor_conf:
                actor_type = actor_conf.get("type", "BP_Character")
                class_name = self._get_class_name(actor_type)
                count = actor_conf.get("count", 1)
                safe_spawns = actor_conf["safe_spawns"]

                for i in range(min(count, len(safe_spawns))):
                    spawn_point = safe_spawns[i]
                    name = f"{actor_type}_{i}"
                    location = [spawn_point["x"], spawn_point["y"], spawn_point["z"]]

                    self.spawn_actor(
                        class_name=class_name,
                        name=name,
                        location=location,
                        rotation=[0, 0, 0],
                    )

            # Legacy format with explicit location
            elif "location" in actor_conf:
                self.spawn_actor(
                    class_name=actor_conf.get("class_name"),
                    name=actor_conf.get("name"),
                    location=actor_conf.get("location"),
                    rotation=actor_conf.get("rotation", [0, 0, 0]),
                    scale=actor_conf.get("scale", None),
                    physics=actor_conf.get("physics", True),
                )

    def _get_class_name(self, actor_type):
        """Convert actor type to UnrealCV class name."""
        type_mapping = {
            "BP_Human": "bp_character_C",
            "BP_Vehicle": "bp_character_C",  # Placeholder - replace with actual vehicle class
            "BP_Prop": "bp_character_C",  # Placeholder - replace with actual prop class
        }
        return type_mapping.get(actor_type, "bp_character_C")

    def spawn_actor(
        self, class_name, name, location, rotation=[0, 0, 0], scale=None, physics=True
    ):
        """
        Spawns a single actor.
        """
        self.logger.info(f"Spawning {name} ({class_name}) at {location}")

        # Use existing new_obj API if compatible, or raw commands
        # gym-unrealcv's new_obj usually handles basic spawn

        # new_obj(self, obj_class_name, obj_name, loc, rot=[0, 0, 0])
        self.client.new_obj(class_name, name, location, rotation)

        if scale:
            self.client.set_obj_scale(name, scale)

        if not physics:
            # gym-unrealcv sets phy=1 by default in new_obj for non-character
            # but we allow overriding
            self.client.set_phy(name, 0)
        else:
            self.client.set_phy(name, 1)

        self.spawned_actors.append(name)
        return name

    def randomize_environment(self, lights=True, texture=True, sky=True):
        """
        Applies domain randomization to the environment using UnrealCV commands.
        """
        # This is a wrapper around potential randomization logic
        # For now, we reuse what might be available in the client or implement custom logic

        # Example light randomization (if supported by client methods)
        if lights and hasattr(self.client, "random_lit"):
            # We need a list of lights. usually in env_config.
            # If we don't have access to env_config directly here, we might skip or require it passed in.
            pass

        # Simple manual randomization example if client helper methods aren't enough
        if sky:
            # Randomize sky light intensity
            intensity = np.random.uniform(1, 5)
            self.client.client.request(
                f"vbp Skylight set_light 1 1 1 {intensity}"
            )  # Assuming Skylight actor exists

    def get_spawned_actors(self):
        return self.spawned_actors
