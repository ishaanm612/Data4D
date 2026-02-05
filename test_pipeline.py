import gym
import gym_unrealcv
import time
import os
import numpy as np
from unrealzoo.core.director import Director
from unrealzoo.core.recorder import MultiViewRecorder
from unrealzoo.utils.ground_detection import load_safe_spawns, get_random_safe_spawn


def test_pipeline(env_id="UnrealAgent-Greek_Island-ContinuousColor-v0"):
    # ===== CONFIGURATION =====
    NUM_ACTORS = 5
    NUM_STATIC_CAMERAS = 5
    ACTOR_SPREAD_RADIUS = 2000  # How far apart actors spawn
    CAMERA_OFFSET_RANGE = 500  # Random offset range for static cameras from actors
    # =========================

    print(f"Initializing Env: {env_id}...")
    env = gym.make(env_id)
    # DO NOT modify resolution - it causes env.reset() to hang
    env.reset()
    client = env.unwrapped.unrealcv

    # Generate unique session ID
    session_id = time.strftime("%Y%m%d_%H%M%S")
    output_base = "demo_output_60fps"
    session_dir = os.path.join(output_base, session_id)
    print(f"Saving data to: {session_dir}")

    # Init Components
    director = Director(client)
    recorder = MultiViewRecorder(client, output_dir=session_dir)

    # 1. Clear scene
    print("Resetting scene...")
    client.client.request("vset /action/clear")

    # Load safe spawn points
    print("Loading safe spawn points from safe_spawns.json...")
    safe_spawns = load_safe_spawns("safe_spawns.json")
    print(f"✓ Loaded {len(safe_spawns)} safe spawn locations")

    print(f"Spawning {NUM_ACTORS} actors at safe locations...")
    spawned = []

    # Select random safe spawn points for each actor
    selected_spawns = np.random.choice(len(safe_spawns), size=NUM_ACTORS, replace=False)

    for i, spawn_idx in enumerate(selected_spawns):
        spawn_point = safe_spawns[spawn_idx]
        x, y, z = spawn_point["x"], spawn_point["y"], spawn_point["z"]

        actor_name = f"Walker_{i+1}"
        client.new_obj("bp_character_C", actor_name, [x, y, z], [0, 0, 0])
        spawned.append(actor_name)
        print(f"  {actor_name} spawned at ({x:.0f}, {y:.0f}, {z:.0f})")

    print(f"✓ Spawned {len(spawned)} agents at safe locations")

    # Allow physics to settle
    print("Waiting 2s for initialization...")
    time.sleep(2.0)

    # 2. Setup Director (NavMesh Movement)
    print("Registering actors for NavMesh movement...")
    for actor in spawned:
        director.register_background_actor(actor)

    # Position camera 0 (viewport) near first actor for observation (one-time setup)
    print("Positioning viewport camera near first actor...")
    try:
        w1_loc = client.get_obj_location(spawned[0])
        cam_x = w1_loc[0] - 400  # Behind
        cam_y = w1_loc[1]
        cam_z = w1_loc[2] + 200  # Above
        client.client.request(f"vset /camera/0/location {cam_x} {cam_y} {cam_z}")
        client.client.request(f"vset /camera/0/rotation -20 0 0")  # Look down slightly
        print(f"Camera positioned at X={cam_x:.1f}, Y={cam_y:.1f}, Z={cam_z:.1f}")
    except Exception as e:
        print(f"Could not position camera: {e}")

    # 3. Setup Cameras
    print("Setting up Virtual Views...")
    recorder.set_resolution(1280, 720)

    # Static Cameras - use safe spawn points with elevated Z
    print(f"Creating {NUM_STATIC_CAMERAS} static cameras...")
    camera_spawn_indices = np.random.choice(
        len(safe_spawns), size=NUM_STATIC_CAMERAS, replace=False
    )

    for i, spawn_idx in enumerate(camera_spawn_indices):
        spawn_point = safe_spawns[spawn_idx]

        # Use safe spawn location with elevated height for camera
        cam_loc = [
            spawn_point["x"],
            spawn_point["y"],
            spawn_point["z"]
            + np.random.uniform(200, 400),  # Additional height for camera view
        ]

        # Find nearest actor to point camera at
        min_dist = float("inf")
        target_actor = spawned[0]
        for actor in spawned:
            actor_loc = client.get_obj_location(actor)
            dist = np.sqrt(
                (actor_loc[0] - cam_loc[0]) ** 2 + (actor_loc[1] - cam_loc[1]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                target_actor = actor

        # Point camera toward nearest actor
        actor_loc = client.get_obj_location(target_actor)
        pitch = -20  # Look down
        yaw = np.degrees(
            np.arctan2(actor_loc[1] - cam_loc[1], actor_loc[0] - cam_loc[0])
        )

        recorder.add_static_view(
            f"static_{i+1}", location=cam_loc, rotation=[pitch, yaw, 0]
        )

    # Per-Actor Views (POV + Follow)
    for actor in spawned:
        recorder.add_actor_pov_view(
            f"{actor}_pov", actor, offset_loc=[20, 0, 80], offset_rot=[0, 0, 0]
        )
        recorder.add_actor_follow_view(
            f"{actor}_follow", actor, distance=300, pitch=-30, yaw=0
        )

    # 4. Run Loop (60 Frames at 30 FPS)
    target_frames = 60
    dt = 1.0 / 30.0

    print(f"Starting Recording for {target_frames} frames at 30 FPS...")

    for i in range(target_frames):
        if i % 10 == 0:
            print(f"Frame {i}/{target_frames}")

        # Step Director (sends NavMesh commands periodically)
        director.step(dt)

        # Capture Frame from all views
        recorder.capture_frame()

        time.sleep(dt)

    print(f"Recording Complete! Data saved to: {session_dir}")
    env.close()


if __name__ == "__main__":
    test_pipeline()
