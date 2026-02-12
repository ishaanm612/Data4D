import cv2
import os
import argparse
import time
import glob
import numpy as np


def replay_data(output_dir, fps=30, step_mode=False):
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} not found.")
        return

    # Check if this is the base directory (demo_output_60fps) or a session directory
    subdirs = [
        d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))
    ]

    # If we see timestamp-like directories (YYYYMMDD_HHMMSS), this is the base dir
    session_dirs = [d for d in subdirs if len(d) == 15 and d[8] == "_"]

    if session_dirs:
        # This is the base directory, select latest session
        session_dirs.sort(reverse=True)
        latest_session = session_dirs[0]
        print(f"Found {len(session_dirs)} sessions. Using latest: {latest_session}")
        session_path = os.path.join(output_dir, latest_session)
    else:
        # This is already a session directory
        session_path = output_dir

    # Find available views within the session
    views = [
        d
        for d in os.listdir(session_path)
        if os.path.isdir(os.path.join(session_path, d))
    ]
    views.sort()

    if not views:
        print(f"No views found in {session_path}")
        return

    print(f"\nFound {len(views)} views:")
    for idx, v in enumerate(views):
        print(f"  {idx}: {v}")

    # Verify valid views with rgb images
    valid_views = []
    for v in views:
        rgb_dir = os.path.join(session_path, v, "rgb")
        if os.path.exists(rgb_dir) and len(os.listdir(rgb_dir)) > 0:
            valid_views.append(v)

    if not valid_views:
        print(f"\nNo valid views with 'rgb' images found in {session_path}")
        return

    if step_mode:
        print(
            f"\nStep mode: [Space]/d next frame, a prev frame, n next view, q quit."
        )
    else:
        print(
            f"\nReplaying {len(valid_views)} views... Press 'q' to quit, 'n' for next view."
        )

    for view_name in valid_views:
        print(f"\nPlaying View: {view_name}")
        rgb_dir = os.path.join(session_path, view_name, "rgb")
        mask_dir = os.path.join(session_path, view_name, "mask")
        depth_dir = os.path.join(session_path, view_name, "depth")
        images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))

        if not images:
            print(f"  No images in {view_name}")
            continue

        frame_stems = [os.path.splitext(os.path.basename(p))[0] for p in images]

        frame_idx = 0
        while frame_idx < len(images):
            frame_stem = frame_stems[frame_idx]
            rgb = cv2.imread(images[frame_idx], cv2.IMREAD_COLOR)
            if rgb is None:
                rgb = np.zeros((360, 640, 3), dtype=np.uint8)

            panel_w = min(640, rgb.shape[1])
            scale = panel_w / float(max(1, rgb.shape[1]))
            panel_h = max(180, int(rgb.shape[0] * scale))
            panel_size = (panel_w, panel_h)  # (w, h)

            rgb_panel = _prepare_panel(rgb, panel_size, "RGB")
            mask_panel = _load_mask_panel(mask_dir, frame_stem, panel_size)
            depth_panel = _load_depth_panel(depth_dir, frame_stem, panel_size)
            info_panel = _build_info_panel(
                panel_size,
                view_name=view_name,
                frame_stem=frame_stem,
                frame_idx=frame_idx,
                total_frames=len(images),
                step_mode=step_mode,
            )

            top_row = np.hstack([rgb_panel, mask_panel])
            bottom_row = np.hstack([depth_panel, info_panel])
            canvas = np.vstack([top_row, bottom_row])

            cv2.imshow("UnrealZoo Replay", canvas)

            if step_mode:
                wait_ms = 0  # Wait indefinitely for keypress
            else:
                wait_ms = int(1000 / fps)
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q"):
                print("Quitting.")
                cv2.destroyAllWindows()
                return
            elif key == ord("n"):
                print("Skipping to next view...")
                break
            elif step_mode:
                if key in (ord(" "), ord("d"), 83):  # space, 'd', or right-arrow
                    frame_idx += 1
                elif key in (ord("a"), 81):  # 'a' or left-arrow
                    frame_idx = max(0, frame_idx - 1)
            else:
                frame_idx += 1

        # Pause briefly between views
        time.sleep(0.5)

    cv2.destroyAllWindows()
    print("\nReplay finished.")


def _prepare_panel(image, panel_size, title):
    panel = cv2.resize(image, panel_size, interpolation=cv2.INTER_LINEAR)
    cv2.putText(
        panel,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    return panel


def _load_mask_panel(mask_dir, frame_stem, panel_size):
    mask_path = os.path.join(mask_dir, f"{frame_stem}.png")
    if not os.path.exists(mask_path):
        return _empty_panel(panel_size, "MASK (missing)")

    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask is None:
        return _empty_panel(panel_size, "MASK (unreadable)")
    return _prepare_panel(mask, panel_size, "MASK")


def _load_depth_panel(depth_dir, frame_stem, panel_size):
    npy_path = os.path.join(depth_dir, f"{frame_stem}.npy")
    png_path = os.path.join(depth_dir, f"{frame_stem}.png")

    depth_vis = None
    if os.path.exists(npy_path):
        try:
            depth = np.load(npy_path, allow_pickle=False)
            depth = np.squeeze(depth)
            if depth.ndim != 2:
                return _empty_panel(panel_size, "DEPTH (shape err)")
            depth_vis = _depth_to_colormap(depth)
        except Exception:
            return _empty_panel(panel_size, "DEPTH (read err)")
    elif os.path.exists(png_path):
        depth_png = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            return _empty_panel(panel_size, "DEPTH (read err)")
        if depth_png.ndim == 2:
            depth_vis = cv2.applyColorMap(
                cv2.normalize(depth_png, None, 0, 255, cv2.NORM_MINMAX).astype(
                    np.uint8
                ),
                cv2.COLORMAP_INFERNO,
            )
        else:
            depth_vis = depth_png
    else:
        return _empty_panel(panel_size, "DEPTH (missing)")

    return _prepare_panel(depth_vis, panel_size, "DEPTH")


def _depth_to_colormap(depth):
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    valid = depth[finite]
    lo = np.percentile(valid, 2.0)
    hi = np.percentile(valid, 98.0)
    if hi <= lo:
        lo = float(valid.min())
        hi = float(valid.max()) + 1e-6

    depth_clip = np.clip(depth, lo, hi)
    depth_norm = ((depth_clip - lo) / (hi - lo + 1e-6) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)


def _empty_panel(panel_size, title):
    panel = np.zeros((panel_size[1], panel_size[0], 3), dtype=np.uint8)
    cv2.putText(
        panel,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    return panel


def _build_info_panel(panel_size, view_name, frame_stem, frame_idx, total_frames, step_mode):
    panel = np.zeros((panel_size[1], panel_size[0], 3), dtype=np.uint8)
    lines = [
        f"View: {view_name}",
        f"Frame: {frame_stem}",
        f"Index: {frame_idx + 1}/{total_frames}",
        "",
    ]
    if step_mode:
        lines.extend(
            [
                "Step controls:",
                "Space/d: next",
                "a: prev",
                "n: next view",
                "q: quit",
            ]
        )
    else:
        lines.extend(
            [
                "Autoplay controls:",
                "n: next view",
                "q: quit",
            ]
        )

    y = 28
    for line in lines:
        cv2.putText(
            panel,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
        )
        y += 26

    return panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay UnrealZoo Recordings")
    parser.add_argument(
        "--dir",
        type=str,
        default="demo_output_60fps",
        help="Output directory to replay",
    )
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS")
    parser.add_argument(
        "--step",
        action="store_true",
        help="Step through frames manually (Space/d=next, a=prev, n=view, q=quit)",
    )
    args = parser.parse_args()

    replay_data(args.dir, args.fps, step_mode=args.step)
