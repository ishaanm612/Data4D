import logging
import time
import numpy as np


class Director:
    """
    Manages the 4D aspect of the scene: temporal evolution, scripted events,
    and stochastic background movements.
    """

    def __init__(self, unreal_client, nav_interval=3.0, nav_loop=0):
        self.client = unreal_client
        self.logger = logging.getLogger(__name__)
        self.scheduled_events = []  # List of dicts with {time, actor, type, params}
        self.background_actors = []  # Actors with stochastic behavior
        self.current_time = 0.0

        # Track last navigation command time for NavMesh
        self.last_nav_time = {}  # {actor_name: last_time}
        self.nav_interval = nav_interval  # Seconds between nav_random; longer = smoother, less disjointed motion
        self.nav_loop = nav_loop  # 0: one-shot random goal, 1: continuous random wandering
        self._continuous_nav_started = {}  # {actor_name: bool}

    def register_background_actor(self, actor_name):
        """
        Register an actor for background/stochastic behavior.
        """
        if actor_name not in self.background_actors:
            self.background_actors.append(actor_name)
            self.last_nav_time[actor_name] = (
                -10.0
            )  # Initialize to allow immediate first command
            self._continuous_nav_started[actor_name] = False
            self.logger.info(f"Registered background actor: {actor_name}")

    def schedule_action(self, timestamp, actor, action_type, params):
        """
        Schedule an action at a specific time.

        Args:
            timestamp (float): Time in simulation to execute
            actor (str): Actor name
            action_type (str): 'move_to', 'animation', 'set_speed', etc.
            params (dict): Action-specific parameters
        """
        self.scheduled_events.append(
            {"time": timestamp, "actor": actor, "type": action_type, "params": params}
        )
        # Sort by time
        self.scheduled_events.sort(key=lambda x: x["time"])

    def step(self, dt):
        """
        Advance the director by dt seconds, executing scheduled events and updating background actors.
        """
        self.current_time += dt

        # Process scheduled events
        while (
            self.scheduled_events
            and self.scheduled_events[0]["time"] <= self.current_time
        ):
            event = self.scheduled_events.pop(0)
            self.execute_event(event)

        # Update background actors
        self.update_background_actors(dt)

    def execute_event(self, event):
        actor = event["actor"]
        atype = event["type"]
        params = event["params"]

        self.logger.info(
            f"Director executing: {atype} on {actor} at t={self.current_time:.2f}"
        )

        if atype == "move_to":
            # params: target_loc=[x,y,z]
            target = params.get("target")
            speed = params.get("speed", 100)
            # Depending on client API. set_move_bp usually takes [angle, speed] or [vx, vy, vz]
            # If we have nav_to_goal
            if hasattr(self.client, "nav_to_goal"):
                self.client.nav_to_goal(actor, target)
            else:
                self.logger.warning(f"nav_to_goal not supported for {actor}")

        elif atype == "animation":
            # params: anim_id='walk'
            anim_id = params.get("anim_id")
            if hasattr(self.client, "set_animation"):
                self.client.set_animation(actor, anim_id)
            else:
                self.client.client.request(f"vbp {actor} play_anim {anim_id}")

        elif atype == "set_speed":
            speed = params.get("speed")
            if hasattr(self.client, "set_max_speed"):
                self.client.set_max_speed(actor, speed)

        elif atype == "stop":
            if hasattr(self.client, "set_move_bp"):
                self.client.set_move_bp(actor, [0, 0])
            self.client.client.request(f"vbp {actor} set_stop")

    def update_background_actors(self, dt):
        """
        Stochastic updates for background actors using NavMesh.
        Sends nav_random commands periodically when loop=0.
        When loop=1 (continuous mode), sends nav_random once per actor and lets
        the engine keep updating targets internally.
        """
        for actor in self.background_actors:
            continuous_mode = bool(self.nav_loop)
            should_send = False

            if continuous_mode:
                # In continuous mode, issue nav_random only once.
                should_send = not self._continuous_nav_started.get(actor, False)
            else:
                # In one-shot mode, re-issue periodically.
                should_send = (
                    self.current_time - self.last_nav_time.get(actor, -10)
                    >= self.nav_interval
                )

            if not should_send:
                continue

            radius = 2500  # Random navigation radius
            try:
                if hasattr(self.client, "nav_random"):
                    self.client.nav_random(actor, radius, self.nav_loop)
                else:
                    self.client.client.request(
                        f"vbp {actor} nav_random {radius} {self.nav_loop}"
                    )
                self.last_nav_time[actor] = self.current_time
                if continuous_mode:
                    self._continuous_nav_started[actor] = True
            except Exception as e:
                self.logger.warning(f"Failed to send nav_random to {actor}: {e}")

    def reset(self):
        self.current_time = 0.0
        self.scheduled_events = []
        self.last_nav_time = {actor: -10.0 for actor in self.background_actors}
        self._continuous_nav_started = {
            actor: False for actor in self.background_actors
        }
        if hasattr(self, "actor_movement"):
            self.actor_movement = {}
