import argparse
import glob
import json
import os

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


def resolve_session_path(output_dir):
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Directory not found: {output_dir}")

    subdirs = [
        d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))
    ]
    session_dirs = [d for d in subdirs if len(d) == 15 and d[8] == "_"]
    if session_dirs:
        session_dirs.sort(reverse=True)
        latest = session_dirs[0]
        print(f"Found {len(session_dirs)} sessions. Using latest: {latest}")
        return os.path.join(output_dir, latest)
    return output_dir


def load_pose_frames(session_path):
    pose_dir = os.path.join(session_path, "poses")
    return sorted(glob.glob(os.path.join(pose_dir, "frame_*.json")))


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_actor_filter(actor_filter):
    if not actor_filter:
        return None
    actors = [a.strip() for a in actor_filter.split(",") if a.strip()]
    return set(actors) if actors else None


def actor_color(actor_id):
    h = abs(hash(actor_id))
    return (
        64 + (h % 160),
        64 + ((h // 97) % 160),
        64 + ((h // 193) % 160),
    )


def extract_transform(actor_data):
    transform = actor_data.get("transform", {})
    position = transform.get("position")
    rotation = transform.get("rotation")

    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return None
    try:
        pos = np.asarray(position[:3], dtype=np.float32)
    except Exception:
        return None

    yaw = 0.0
    if isinstance(rotation, (list, tuple)) and len(rotation) >= 2:
        try:
            yaw = float(rotation[1])
        except (TypeError, ValueError):
            yaw = 0.0

    return {"pos": pos, "yaw": yaw}


def extract_joint_points(actor_data):
    joints_data = actor_data.get("joints", {})
    points = {}

    positions = joints_data.get("positions", {})
    if isinstance(positions, dict):
        for name, xyz in positions.items():
            if isinstance(xyz, (list, tuple)) and len(xyz) >= 3:
                try:
                    points[str(name)] = np.asarray(xyz[:3], dtype=np.float32)
                except Exception:
                    continue
    return points


def load_canonical_mesh(session_path, actor_data, mesh_cache, max_points):
    mesh_meta = actor_data.get("canonical_mesh", {})
    rel_path = mesh_meta.get("path")
    if not rel_path:
        return None, None

    abs_path = os.path.join(session_path, rel_path)
    if abs_path in mesh_cache:
        return mesh_cache[abs_path], int(mesh_meta.get("frame_captured", 0))

    if not os.path.exists(abs_path):
        mesh_cache[abs_path] = None
        return None, int(mesh_meta.get("frame_captured", 0))

    try:
        arr = np.load(abs_path, allow_pickle=False)
    except Exception:
        mesh_cache[abs_path] = None
        return None, int(mesh_meta.get("frame_captured", 0))

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        mesh_cache[abs_path] = None
        return None, int(mesh_meta.get("frame_captured", 0))

    arr = arr[:, :3]
    if max_points > 0 and arr.shape[0] > max_points:
        stride = int(np.ceil(arr.shape[0] / float(max_points)))
        arr = arr[::stride]

    mesh_cache[abs_path] = arr
    return arr, int(mesh_meta.get("frame_captured", 0))


def get_reference_transform(session_path, actor_id, frame_captured, ref_transform_cache):
    key = (actor_id, frame_captured)
    if key in ref_transform_cache:
        return ref_transform_cache[key]

    ref_path = os.path.join(session_path, "poses", f"frame_{frame_captured:06d}.json")
    if not os.path.exists(ref_path):
        ref_transform_cache[key] = None
        return None

    try:
        frame = read_json(ref_path)
        actor_data = frame.get("actors", {}).get(actor_id, {})
        transform = extract_transform(actor_data)
    except Exception:
        transform = None

    ref_transform_cache[key] = transform
    return transform


def transform_mesh(mesh_ref_world, ref_transform, cur_transform):
    if mesh_ref_world is None:
        return None
    if ref_transform is None or cur_transform is None:
        return mesh_ref_world

    ref_pos = ref_transform["pos"]
    cur_pos = cur_transform["pos"]
    yaw_delta = np.deg2rad(cur_transform["yaw"] - ref_transform["yaw"])

    c = np.cos(yaw_delta)
    s = np.sin(yaw_delta)

    local = mesh_ref_world - ref_pos[None, :]
    x_new = c * local[:, 0] - s * local[:, 1]
    y_new = s * local[:, 0] + c * local[:, 1]
    z_new = local[:, 2]
    transformed = np.stack([x_new, y_new, z_new], axis=1) + cur_pos[None, :]
    return transformed


def collect_global_bounds(frame_paths, actor_filter):
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    has_any = False

    for frame_path in frame_paths:
        try:
            frame = read_json(frame_path)
        except Exception:
            continue

        actors = frame.get("actors", {})
        if not isinstance(actors, dict):
            continue

        for actor_id, actor_data in actors.items():
            if actor_filter and actor_id not in actor_filter:
                continue

            transform = extract_transform(actor_data)
            if transform is not None:
                mins = np.minimum(mins, transform["pos"])
                maxs = np.maximum(maxs, transform["pos"])
                has_any = True

            joints = extract_joint_points(actor_data)
            for xyz in joints.values():
                mins = np.minimum(mins, xyz)
                maxs = np.maximum(maxs, xyz)
                has_any = True

    if not has_any:
        return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])

    # Expand a bit so body mesh around roots is visible.
    pad = np.array([250.0, 250.0, 200.0], dtype=np.float64)
    mins = mins - pad
    maxs = maxs + pad

    for i in range(3):
        if abs(maxs[i] - mins[i]) < 1e-6:
            mins[i] -= 0.5
            maxs[i] += 0.5
    return mins, maxs


def project_points(points_xyz, axis_u, axis_v, mins, maxs, width, height, margin=28):
    if points_xyz is None or len(points_xyz) == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    u = points_xyz[:, axis_u]
    v = points_xyz[:, axis_v]
    un = (u - mins[axis_u]) / (maxs[axis_u] - mins[axis_u] + 1e-9)
    vn = (v - mins[axis_v]) / (maxs[axis_v] - mins[axis_v] + 1e-9)

    x = (margin + un * (width - 2 * margin)).astype(np.int32)
    y = (height - margin - vn * (height - 2 * margin)).astype(np.int32)
    return x, y


def draw_projection_panel(
    title,
    axis_u,
    axis_v,
    mins,
    maxs,
    frame_actors,
    show_mesh=True,
    width=540,
    height=540,
    label_joints=False,
):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (70, 70, 70), 1)
    cv2.putText(panel, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)

    axis_name = ["X", "Y", "Z"]
    cv2.putText(
        panel,
        f"{axis_name[axis_u]} vs {axis_name[axis_v]}",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 160, 160),
        1,
    )

    legend_y = 76
    for actor in frame_actors:
        actor_id = actor["id"]
        joints = actor["joints"]
        mesh = actor["mesh"]
        color = actor_color(actor_id)

        cv2.circle(panel, (14, legend_y - 4), 5, color, -1)
        cv2.putText(panel, actor_id, (26, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        legend_y += 18
        if legend_y > height - 12:
            break

        if show_mesh and mesh is not None and len(mesh) > 0:
            x, y = project_points(mesh, axis_u, axis_v, mins, maxs, width, height)
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            x = x[valid]
            y = y[valid]
            mesh_color = (
                max(20, int(color[0] * 0.45)),
                max(20, int(color[1] * 0.45)),
                max(20, int(color[2] * 0.45)),
            )
            panel[y, x] = mesh_color

        if joints:
            joint_xyz = np.stack(list(joints.values()), axis=0)
            x, y = project_points(joint_xyz, axis_u, axis_v, mins, maxs, width, height)
            for idx, (jx, jy) in enumerate(zip(x, y)):
                cv2.circle(panel, (int(jx), int(jy)), 3, color, -1)
                if label_joints:
                    name = list(joints.keys())[idx]
                    cv2.putText(panel, name, (int(jx) + 3, int(jy) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

    return panel


def draw_info_panel(frame_stem, frame_idx, total_frames, timestamp, actor_count, mesh_count, step_mode, paused, show_mesh):
    width, height = 540, 540
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    lines = [
        "Pose + Mesh Visualizer",
        "",
        f"Frame: {frame_stem}",
        f"Index: {frame_idx + 1}/{total_frames}",
        f"Timestamp: {timestamp:.3f}",
        f"Actors in frame: {actor_count}",
        f"Actors with mesh: {mesh_count}",
        f"Show mesh: {show_mesh}",
        f"Paused: {paused}",
        "",
        "Controls:",
        "q: quit",
        "space: pause/resume",
        "d/right: next frame",
        "a/left: previous frame",
        "m: toggle mesh",
    ]
    if step_mode:
        lines.append("Step mode is enabled")

    y = 34
    for i, line in enumerate(lines):
        color = (220, 220, 220) if i != 0 else (180, 240, 180)
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 if i == 0 else 0.52, color, 1 if i else 2)
        y += 26
    return panel


def build_frame_actor_payload(session_path, frame, actor_filter_set, mesh_cache, ref_transform_cache, mesh_max_points):
    actors = frame.get("actors", {})
    payload = []
    if not isinstance(actors, dict):
        return payload

    for actor_id, actor_data in actors.items():
        if actor_filter_set and actor_id not in actor_filter_set:
            continue

        cur_transform = extract_transform(actor_data)
        joints = extract_joint_points(actor_data)

        if not joints and cur_transform is not None:
            joints = {"root": cur_transform["pos"]}

        mesh_ref, frame_captured = load_canonical_mesh(
            session_path, actor_data, mesh_cache, mesh_max_points
        )
        mesh_world = None
        if mesh_ref is not None:
            ref_transform = get_reference_transform(
                session_path, actor_id, frame_captured, ref_transform_cache
            )
            mesh_world = transform_mesh(mesh_ref, ref_transform, cur_transform)

        payload.append({"id": actor_id, "joints": joints, "mesh": mesh_world})

    return payload


def visualize_poses(output_dir, fps=10, step_mode=False, actor_filter=None, label=False, mesh_max_points=2500, hide_mesh=False):
    session_path = resolve_session_path(output_dir)
    frame_paths = load_pose_frames(session_path)
    if not frame_paths:
        print(f"No pose frames found in: {os.path.join(session_path, 'poses')}")
        print("Run test_pipeline with pose capture enabled to generate poses/frame_*.json.")
        return

    actor_filter_set = parse_actor_filter(actor_filter)
    mins, maxs = collect_global_bounds(frame_paths, actor_filter_set)

    print(f"Loaded {len(frame_paths)} pose frames from: {session_path}")
    print(f"Bounds | X:[{mins[0]:.1f}, {maxs[0]:.1f}] Y:[{mins[1]:.1f}, {maxs[1]:.1f}] Z:[{mins[2]:.1f}, {maxs[2]:.1f}]")
    if actor_filter_set:
        print(f"Actor filter: {sorted(actor_filter_set)}")

    mesh_cache = {}
    ref_transform_cache = {}
    frame_idx = 0
    paused = bool(step_mode)
    show_mesh = not hide_mesh
    wait_ms = max(1, int(1000 / max(1, fps)))

    while 0 <= frame_idx < len(frame_paths):
        frame_path = frame_paths[frame_idx]
        frame_stem = os.path.splitext(os.path.basename(frame_path))[0]
        try:
            frame = read_json(frame_path)
        except Exception as exc:
            print(f"Failed to read {frame_path}: {exc}")
            frame_idx += 1
            continue

        frame_actors = build_frame_actor_payload(
            session_path, frame, actor_filter_set, mesh_cache, ref_transform_cache, mesh_max_points
        )
        mesh_count = sum(1 for a in frame_actors if a["mesh"] is not None)

        panel_xy = draw_projection_panel("Top View", 0, 1, mins, maxs, frame_actors, show_mesh=show_mesh, label_joints=label)
        panel_xz = draw_projection_panel("Side View", 0, 2, mins, maxs, frame_actors, show_mesh=show_mesh, label_joints=label)
        panel_yz = draw_projection_panel("Front View", 1, 2, mins, maxs, frame_actors, show_mesh=show_mesh, label_joints=label)
        info = draw_info_panel(
            frame_stem=frame_stem,
            frame_idx=frame_idx,
            total_frames=len(frame_paths),
            timestamp=float(frame.get("timestamp", 0.0)),
            actor_count=len(frame_actors),
            mesh_count=mesh_count,
            step_mode=step_mode,
            paused=paused,
            show_mesh=show_mesh,
        )

        canvas = np.vstack([np.hstack([panel_xy, panel_xz]), np.hstack([panel_yz, info])])
        cv2.imshow("UnrealZoo Pose + Mesh Visualizer", canvas)

        key = cv2.waitKey(0 if paused else wait_ms) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord("m"):
            show_mesh = not show_mesh
            continue
        if key in (ord("a"), 81):  # left
            frame_idx = max(0, frame_idx - 1)
            continue
        if key in (ord("d"), 83):  # right
            frame_idx = min(len(frame_paths) - 1, frame_idx + 1)
            continue

        if not paused:
            frame_idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if cv2 is None:
        raise SystemExit(
            "OpenCV (cv2) is required for visualization. "
            "Install it in your runtime env, e.g. `pip install opencv-python`."
        )

    parser = argparse.ArgumentParser(description="Visualize walker mesh and joint positions from poses/*.json")
    parser.add_argument("--dir", type=str, default="demo_output_60fps", help="Base output directory or specific session directory")
    parser.add_argument("--fps", type=int, default=10, help="Playback FPS in autoplay mode")
    parser.add_argument("--step", action="store_true", help="Start paused and step through frames manually")
    parser.add_argument("--actors", type=str, default="", help="Comma-separated actor IDs to visualize (default: all)")
    parser.add_argument("--label-joints", action="store_true", help="Draw joint name labels (can get crowded)")
    parser.add_argument("--mesh-max-points", type=int, default=2500, help="Max mesh points per actor to render")
    parser.add_argument("--hide-mesh", action="store_true", help="Start with mesh hidden (toggle with 'm')")
    args = parser.parse_args()

    visualize_poses(
        output_dir=args.dir,
        fps=args.fps,
        step_mode=args.step,
        actor_filter=args.actors,
        label=args.label_joints,
        mesh_max_points=max(0, int(args.mesh_max_points)),
        hide_mesh=args.hide_mesh,
    )
