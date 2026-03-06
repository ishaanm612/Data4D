import gym
import gym_unrealcv
import time
import numpy as np


def demo_spawn_and_move():
    # Initialize the environment
    # using a simple environment ID from the examples
    env_id = "UnrealAgent-MiddleEast-ContinuousColor-v0"
    env = gym.make(env_id)
    env.reset()

    # Access the underlying UnrealCV client interface
    # The gym wrapper might wrap the env multiple times, so we might need to go deeper
    # But usually env.unwrapped gives access to the base env
    unreal = env.unwrapped.unrealcv

    print("Environment Initialized.")

    # --- 1. Spawning a New Model ---
    # We can spawn different types of objects.
    # 'bp_character_C' is often used for agents/players. 'Cube' is a basic shape.
    # Note: The class name depends on what is available in the compiled binary.

    new_agent_name = "Agent_Spawned"
    # Location: [x, y, z]
    spawn_loc = [0, 0, 200]
    # Rotation: [pitch, yaw, roll]
    spawn_rot = [0, 0, 0]

    # Depending on the binary, specific blueprint classes are available.
    # 'bp_character_C' is a common one for characters in UnrealZoo.
    # If that fails, we can try a basic 'Cube' for demonstration if available or just proceed with what we have.
    print(f"Spawning {new_agent_name} at {spawn_loc}...")

    # new_obj(obj_class_name, obj_name, loc, rot)
    # Trying to spawn a character.
    # Note: If 'bp_character_C' is not in the binary, this might not show up visually or default to something else.
    unreal.new_obj("bp_character_C", new_agent_name, spawn_loc, spawn_rot)

    time.sleep(2)  # Wait for spawn

    # --- 2. Moving the Model Directly (Teleportation) ---
    print(f"Teleporting {new_agent_name}...")
    new_loc = [200, 200, 200]
    unreal.set_obj_location(new_agent_name, new_loc)

    time.sleep(2)

    # --- 3. Moving the Model via Velocity/Physics (Set Move) ---
    print(f"Moving {new_agent_name} with velocity...")
    # set_move_bp(player, params)
    # params: [angle, speed] for 2D movement usually
    unreal.set_move_bp(new_agent_name, [45, 200])  # Move at angle 45, speed 200

    # Run a short loop to let it move
    for _ in range(20):
        actions = [space.sample() for space in env.action_space]
        env.step(actions)  # Step the environment to keep physics running
        # Optional: Print location to track movement
        # pos = unreal.get_obj_location(new_agent_name)
        # print(f"Current pos: {pos}")
        time.sleep(0.1)

    # Stop movement
    unreal.set_move_bp(new_agent_name, [0, 0])
    print("Movement stopped.")

    # --- 4. Navigation (if supported) ---
    target_pos = [500, 500, 200]
    print(f"Navigating {new_agent_name} to {target_pos}...")
    unreal.nav_to_goal(new_agent_name, target_pos)

    for _ in range(50):
        actions = [space.sample() for space in env.action_space]
        env.step(actions)
        time.sleep(0.1)

    print("Demo complete.")
    env.close()


if __name__ == "__main__":
    demo_spawn_and_move()
