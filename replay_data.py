import cv2
import os
import argparse
import time
import glob


def replay_data(output_dir, fps=30):
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

        for img_path in images:
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
                f"Frame: {frame_num}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("UnrealZoo Replay", frame)

            # Wait time based on FPS
            wait_ms = int(1000 / fps)
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q"):
                print("Quitting.")
                cv2.destroyAllWindows()
                return
            elif key == ord("n"):
                print("Skipping to next view...")
                break

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
    args = parser.parse_args()

    replay_data(args.dir, args.fps)
