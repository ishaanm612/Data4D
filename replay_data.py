import cv2
import os
import argparse
import time
import glob


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
        image_folder = os.path.join(session_path, view_name, "rgb")
        images = sorted(glob.glob(os.path.join(image_folder, "*.png")))

        if not images:
            print(f"  No images in {view_name}")
            continue

        frame_idx = 0
        while frame_idx < len(images):
            img_path = images[frame_idx]
            frame = cv2.imread(img_path)

            # Overlay text
            cv2.putText(
                frame,
                f"View: {view_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            frame_num = os.path.basename(img_path).split(".")[0]
            cv2.putText(
                frame,
                f"Frame: {frame_num} ({frame_idx + 1}/{len(images)})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            if step_mode:
                cv2.putText(
                    frame,
                    "[Space]/d next | a prev | n view | q quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                (128, 128, 128),
                1,
                )

            cv2.imshow("UnrealZoo Replay", frame)

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
