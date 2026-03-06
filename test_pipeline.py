import gym
import gym_unrealcv
import time
import os
import numpy as np
from gym_unrealcv.envs.wrappers import configUE
from data4d.core.director import Director
from data4d.core.recorder import MultiViewRecorder
from data4d.utils.ground_detection import load_safe_spawns, get_random_safe_spawn


def test_pipeline(env_id="UnrealAgent-MiddleEast-ContinuousColor-v0"):
    # ===== CONFIGURATION =====
    NUM_ACTORS = 5
    NUM_STATIC_CAMERAS = 5
    ACTOR_SPREAD_RADIUS = 2000  # How far apart actors spawn
    CAMERA_OFFSET_RANGE = 500  # Random offset range for static cameras from actors
    NAV_LOOP = 1  # 0: one-shot random target, 1: continuous random wandering
    # Use 1280x720 to avoid reset hang; Full HD (1920x1080) can stall during init observation
    WINDOW_RESOLUTION = (1280, 720)
    # =========================

    print(f"Initializing Env: {env_id}...")
    env = gym.make(env_id)
    # sleep_time: wait for Unreal to start before connecting.
    # first_obs_delay: wait before first image fetch (Unreal shader compile, etc.).
    env = configUE.ConfigUEWrapper(
        env,
        offscreen=True,
        resolution=WINDOW_RESOLUTION,
        sleep_time=10,
        first_obs_delay=3,
    )
    # Use ensure_launched() to skip initial observation fetch (avoids hang on first render).
    # test_pipeline clears the scene and spawns its own agents, so the default obs isn't needed.
    print("Launching env (connect only, no observation fetch)...")
    client = env.unwrapped.ensure_launched()
    print("Env ready.")

    # Generate unique session ID
    session_id = time.strftime("%Y%m%d_%H%M%S")
    output_base = "demo_output_60fps"
    session_dir = os.path.join(output_base, session_id)
    print(f"Saving data to: {session_dir}")

    # Init Components
    print("Initializing Director and Recorder...")
    director = Director(client, nav_loop=NAV_LOOP)
    # skip_depth=True: depth capture often hangs with UnrealCV; RGB/mask still captured
    recorder = MultiViewRecorder(client, output_dir=session_dir, skip_depth=False)

    # 1. Clear scene (removes default env agents so we can spawn our own)
    print("Resetting scene (vset /action/clear)...")
    try:
        client.client.request("vset /action/clear")
        print("  Scene clear done. Waiting 2s for Unreal to settle...")
        time.sleep(2.0)
    except Exception as e:
        print(f"  Warning: scene clear failed ({e}). Continuing...")

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
        print(f"  Spawning {actor_name}...", end=" ", flush=True)
        try:
            client.new_obj("bp_character_C", actor_name, [x, y, z], [0, 0, 0])
            spawned.append(actor_name)
            print(f"done at ({x:.0f}, {y:.0f}, {z:.0f})")
        except Exception as e:
            print(f"FAILED: {e}")
            raise

    print(f"✓ Spawned {len(spawned)} agents at safe locations")

    # Allow physics to settle
    print("Waiting 2s for initialization...")
    time.sleep(2.0)

    # 2. Setup Director (NavMesh Movement)
    print("Registering actors for NavMesh movement...")
    for actor in spawned:
        director.register_background_actor(actor)

    # Save one canonical mesh per actor + per-frame joints/poses (if supported by UnrealCV/UE plugin)
    recorder.configure_pose_capture(
        actor_ids=spawned,
        save_canonical_mesh=True,
        save_joints_per_frame=True,
        joint_names=None,  # auto-discover if the backend exposes joint names
        fail_on_joint_capture_error=True,
    )

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
    recorder.set_resolution(WINDOW_RESOLUTION[0], WINDOW_RESOLUTION[1])

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
    # offset_loc=[forward, right, up] in actor local frame; 80 forward to avoid head clipping
    for actor in spawned:
        recorder.add_actor_pov_view(
            f"{actor}_pov", actor, offset_loc=[80, 0, 80], offset_rot=[0, 0, 0]
        )
        recorder.add_actor_follow_view(
            f"{actor}_follow", actor, distance=300, pitch=-30, yaw=0
        )

    # 4. Run Loop (600 Frames at 30 FPS)
    target_frames = 30
    dt = 1.0 / 30.0

    print(f"Starting Recording for {target_frames} frames at 30 FPS...")
    session_start = time.time()

    for i in range(target_frames):
        frame_start = time.time()
        print(f"  Frame {i + 1}/{target_frames}: director.step...", end=" ", flush=True)

        # Step Director and let sim advance BEFORE capture (so actors actually move)
        director.step(dt)
        print("resume...", end=" ", flush=True)
        client.client.request("vset /action/game/resume")
        time.sleep(dt)

        # Pause sim for consistent multi-camera capture
        print("pause...", end=" ", flush=True)
        client.client.request("vset /action/game/pause")

        # Capture Frame from all views (sim frozen)
        print("capture...", end=" ", flush=True)
        recorder.capture_frame()
        capture_elapsed = time.time() - frame_start

        total_elapsed = time.time() - session_start
        pct = 100 * (i + 1) / target_frames
        print(f"done | capture: {capture_elapsed:.1f}s | total: {total_elapsed:.1f}s")

    total_time = time.time() - session_start
    print(
        f"Recording Complete! {target_frames} frames in {total_time:.1f}s | Data saved to: {session_dir}"
    )
    env.close()


if __name__ == "__main__":
    test_pipeline()
