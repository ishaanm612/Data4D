import logging
import os
import json
import numpy as np
import time
from pathlib import Path
import io
import cv2


class MultiViewRecorder:
    """
    Handles multi-camera synchronized recording of various modalities:
    RGB, depth, optical flow, segmentation masks, camera intrinsics/extrinsics.
    """

    def __init__(self, unreal_client, output_dir, camera_configs=None):
        """
        Args:
            unreal_client: UnrealCV API client
            output_dir: Root directory for saving recordings
            camera_configs: Optional list of dicts with camera specs (for new API):
                [{
                    'id': 'cam0',
                    'position': [x, y, z],
                    'rotation': [pitch, yaw, roll],
                    'fov': 90,
                    'resolution': (1920, 1080),
                    'type': 'exo'  # or 'ego'
                }, ...]
                If None, cameras can be added dynamically with add_*_view methods
        """
        self.unreal = unreal_client
        self.output_dir = Path(output_dir)
        self.camera_configs = camera_configs if camera_configs is not None else []
        self.frame_idx = 0

        # Initialize logger early
        self.logger = logging.getLogger(__name__)

        # Legacy API support
        self.views = []  # For dynamically added views

        # Create directory structure
        self._init_directories()

        # Store camera metadata if provided
        if camera_configs:
            self._save_camera_metadata()

    def set_resolution(self, width: int, height: int):
        """
        Set the capture resolution for all cameras.

        Args:
            width: Resolution width
            height: Resolution height
        """
        self.logger.info(f"Setting resolution to {width}x{height}")
        result = self.unreal.client.request(f"vset /camera/0/size {width} {height}")
        self.logger.info(f"Resolution set result: {result}")

    def _init_directories(self):
        """Create directory structure for each camera and modality."""
        for cam_config in self.camera_configs:
            cam_id = cam_config["id"]
            cam_dir = self.output_dir / cam_id

            # Create subdirectories for each modality
            for modality in ["rgb", "depth", "mask", "flow", "metadata"]:
                (cam_dir / modality).mkdir(parents=True, exist_ok=True)

        # Create scene-level directories
        (self.output_dir / "scene_metadata").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "poses").mkdir(parents=True, exist_ok=True)

    def _save_camera_metadata(self):
        """Save camera intrinsics and initial extrinsics."""
        metadata = {"cameras": []}

        for cam_config in self.camera_configs:
            cam_id = cam_config["id"]
            resolution = cam_config["resolution"]
            fov = cam_config["fov"]

            # Compute intrinsics (simplified pinhole model)
            fx = fy = (resolution[0] / 2.0) / np.tan(np.deg2rad(fov / 2.0))
            cx, cy = resolution[0] / 2.0, resolution[1] / 2.0

            intrinsics = {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": resolution[0],
                "height": resolution[1],
                "fov": fov,
            }

            cam_meta = {
                "id": cam_id,
                "type": cam_config.get("type", "exo"),
                "intrinsics": intrinsics,
                "initial_position": cam_config["position"],
                "initial_rotation": cam_config["rotation"],
            }

            metadata["cameras"].append(cam_meta)

        # Save to file
        meta_path = self.output_dir / "camera_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved camera metadata to {meta_path}")

    def setup_cameras(self):
        """Position all cameras according to their configurations."""
        for cam_config in self.camera_configs:
            cam_id = cam_config["id"]

            # Set camera pose
            self.unreal.set_location(cam_id, cam_config["position"])
            self.unreal.set_rotation(cam_id, cam_config["rotation"])

            # Set camera FOV if supported
            if hasattr(self.unreal, "set_fov"):
                self.unreal.set_fov(cam_id, cam_config["fov"])

            self.logger.info(f"Camera {cam_id} positioned at {cam_config['position']}")

    def capture_frame(self, timestamp=None):
        """
        Capture all modalities from all cameras for the current frame.

        Args:
            timestamp: Optional timestamp for this frame

        Returns:
            dict: Frame metadata with paths to saved files
        """
        if timestamp is None:
            timestamp = time.time()

        frame_meta = {
            "frame_idx": self.frame_idx,
            "timestamp": timestamp,
            "cameras": {},
        }

        for cam_config in self.camera_configs:
            cam_id = cam_config["id"]
            cam_dir = self.output_dir / cam_id

            cam_frame_meta = {}

            # Capture RGB
            rgb_img = self.unreal.get_img(cam_id, "lit")
            if rgb_img is not None:
                rgb_path = cam_dir / "rgb" / f"frame_{self.frame_idx:06d}.png"
                self.unreal.save_img(rgb_img, str(rgb_path))
                cam_frame_meta["rgb"] = str(rgb_path.relative_to(self.output_dir))

            # Capture Depth
            depth_img = self.unreal.get_img(cam_id, "depth")
            if depth_img is not None:
                depth_path = cam_dir / "depth" / f"frame_{self.frame_idx:06d}.png"
                self.unreal.save_img(depth_img, str(depth_path))
                cam_frame_meta["depth"] = str(depth_path.relative_to(self.output_dir))

            # Capture Segmentation Mask
            mask_img = self.unreal.get_img(cam_id, "object_mask")
            if mask_img is not None:
                mask_path = cam_dir / "mask" / f"frame_{self.frame_idx:06d}.png"
                self.unreal.save_img(mask_img, str(mask_path))
                cam_frame_meta["mask"] = str(mask_path.relative_to(self.output_dir))

            # Capture Optical Flow (if supported in v1.0.4+)
            if hasattr(self.unreal, "get_optical_flow"):
                flow = self.unreal.get_optical_flow(cam_id)
                if flow is not None:
                    flow_path = cam_dir / "flow" / f"frame_{self.frame_idx:06d}.npy"
                    np.save(flow_path, flow)
                    cam_frame_meta["flow"] = str(flow_path.relative_to(self.output_dir))

            # Get camera pose
            position = self.unreal.get_location(cam_id)
            rotation = self.unreal.get_rotation(cam_id)

            cam_frame_meta["pose"] = {"position": position, "rotation": rotation}

            frame_meta["cameras"][cam_id] = cam_frame_meta

        # Save frame metadata
        frame_meta_path = (
            self.output_dir / "scene_metadata" / f"frame_{self.frame_idx:06d}.json"
        )
        with open(frame_meta_path, "w") as f:
            json.dump(frame_meta, f, indent=2)

        self.frame_idx += 1
        return frame_meta

    def record_sequence(self, duration_frames, fps=30):
        """
        Record a sequence of frames.

        Args:
            duration_frames: Number of frames to record
            fps: Target frames per second

        Returns:
            dict: Sequence metadata
        """
        sequence_meta = {"frames": [], "fps": fps, "total_frames": duration_frames}

        frame_time = 1.0 / fps

        for i in range(duration_frames):
            start_time = time.time()

            # Capture frame
            frame_meta = self.capture_frame()
            sequence_meta["frames"].append(frame_meta)

            # Log progress
            if (i + 1) % 10 == 0:
                self.logger.info(f"Recorded frame {i + 1}/{duration_frames}")

            # Maintain frame rate
            elapsed = time.time() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

        # Save sequence metadata
        seq_meta_path = self.output_dir / "sequence_metadata.json"
        with open(seq_meta_path, "w") as f:
            json.dump(sequence_meta, f, indent=2)

        self.logger.info(f"Sequence recording complete: {duration_frames} frames")
        return sequence_meta

    def reset(self):
        """Reset frame counter for a new recording session."""
        self.frame_idx = 0
        self.logger.info("Recorder reset")

    # =================================================================
    # Legacy API for backward compatibility with test_pipeline.py
    # =================================================================

    def add_static_view(self, view_id, location, rotation):
        """
        Add a static camera view (legacy API).

        Args:
            view_id: Unique identifier for this view
            location: [x, y, z] camera position
            rotation: [pitch, yaw, roll] camera rotation
        """
        # Assign a unique camera ID for this view (start from 1, skip 0 which is viewport)
        cam_id = len(self.views) + 1

        view = {
            "id": view_id,
            "type": "static",
            "cam_id": cam_id,
            "location": location,
            "rotation": rotation,
        }
        self.views.append(view)

        # Set up the virtual camera immediately
        loc = location
        rot = rotation
        self.unreal.client.request(
            f"vset /camera/{cam_id}/location {loc[0]} {loc[1]} {loc[2]}"
        )
        self.unreal.client.request(
            f"vset /camera/{cam_id}/rotation {rot[0]} {rot[1]} {rot[2]}"
        )

        # Create output directory for this view
        view_dir = self.output_dir / view_id
        for modality in ["rgb", "depth", "mask", "metadata"]:
            (view_dir / modality).mkdir(parents=True, exist_ok=True)

    def add_actor_pov_view(
        self, view_id, actor_id, offset_loc=[0, 0, 0], offset_rot=[0, 0, 0]
    ):
        """
        Add a first-person POV camera attached to an actor (legacy API).

        Args:
            view_id: Unique identifier for this view
            actor_id: Actor to attach camera to
            offset_loc: [x, y, z] offset from actor location
            offset_rot: [pitch, yaw, roll] rotation offset
        """
        cam_id = len(self.views) + 1

        view = {
            "id": view_id,
            "type": "actor_pov",
            "cam_id": cam_id,
            "actor_id": actor_id,
            "offset_loc": offset_loc,
            "offset_rot": offset_rot,
        }
        self.views.append(view)

        # Create output directory
        view_dir = self.output_dir / view_id
        for modality in ["rgb", "depth", "mask", "metadata"]:
            (view_dir / modality).mkdir(parents=True, exist_ok=True)

    def add_actor_follow_view(self, view_id, actor_id, distance=300, pitch=-30, yaw=0):
        """
        Add a third-person follow camera for an actor (legacy API).

        Args:
            view_id: Unique identifier for this view
            actor_id: Actor to follow
            distance: Distance behind actor
            pitch: Camera pitch angle
            yaw: Camera yaw offset
        """
        cam_id = len(self.views) + 1

        view = {
            "id": view_id,
            "type": "actor_follow",
            "cam_id": cam_id,
            "actor_id": actor_id,
            "distance": distance,
            "pitch": pitch,
            "yaw": yaw,
        }
        self.views.append(view)

        # Create output directory
        view_dir = self.output_dir / view_id
        for modality in ["rgb", "depth", "mask", "metadata"]:
            (view_dir / modality).mkdir(parents=True, exist_ok=True)

    def capture_frame(self):
        """
        Capture a single frame from all views (legacy API).

        Uses virtual cameras (cam_id 1+) to avoid moving the viewport camera (cam_id 0).
        """
        import cv2

        # Capture from all legacy views using their assigned camera IDs
        for view in self.views:
            view_dir = self.output_dir / view["id"]
            cam_id = view["cam_id"]

            try:
                # Update dynamic camera positions (POV and Follow cameras)
                if view["type"] == "actor_pov":
                    # Get actor location directly using UnrealCV command
                    # This works for both registered and safe-spawn actors
                    actor_id = view["actor_id"]
                    loc_str = self.unreal.client.request(
                        f"vget /object/{actor_id}/location"
                    )

                    if loc_str and loc_str != "error":
                        try:
                            actor_loc = [float(x) for x in loc_str.split()]
                            offset = view["offset_loc"]
                            loc = [actor_loc[i] + offset[i] for i in range(3)]
                            rot = view["offset_rot"]
                            self.unreal.client.request(
                                f"vset /camera/{cam_id}/location {loc[0]} {loc[1]} {loc[2]}"
                            )
                            self.unreal.client.request(
                                f"vset /camera/{cam_id}/rotation {rot[0]} {rot[1]} {rot[2]}"
                            )
                        except (ValueError, IndexError) as e:
                            if self.frame_idx == 0:
                                print(
                                    f"✗ [{view['id']}] Failed to get actor location: {e}"
                                )
                    elif self.frame_idx == 0:
                        print(f"✗ [{view['id']}] Actor '{actor_id}' not found")

                elif view["type"] == "actor_follow":
                    # Get actor location and rotation directly using UnrealCV commands
                    actor_id = view["actor_id"]
                    loc_str = self.unreal.client.request(
                        f"vget /object/{actor_id}/location"
                    )
                    rot_str = self.unreal.client.request(
                        f"vget /object/{actor_id}/rotation"
                    )

                    if (
                        loc_str
                        and loc_str != "error"
                        and rot_str
                        and rot_str != "error"
                    ):
                        try:
                            actor_loc = [float(x) for x in loc_str.split()]
                            actor_rot = [float(x) for x in rot_str.split()]

                            # Calculate follow position
                            import math

                            yaw_rad = math.radians(actor_rot[1] + view["yaw"])
                            distance = view["distance"]

                            cam_x = actor_loc[0] - distance * math.cos(yaw_rad)
                            cam_y = actor_loc[1] - distance * math.sin(yaw_rad)
                            cam_z = actor_loc[2] + 100  # Slight elevation

                            self.unreal.client.request(
                                f"vset /camera/{cam_id}/location {cam_x} {cam_y} {cam_z}"
                            )
                            self.unreal.client.request(
                                f"vset /camera/{cam_id}/rotation {view['pitch']} {actor_rot[1] + view['yaw']} 0"
                            )
                        except (ValueError, IndexError) as e:
                            if self.frame_idx == 0:
                                print(
                                    f"✗ [{view['id']}] Failed to parse actor data: {e}"
                                )
                    elif self.frame_idx == 0:
                        print(f"✗ [{view['id']}] Actor '{actor_id}' not found")

                # Static cameras don't need position updates (set during add_static_view)

                # Capture Modalities: RGB, Depth, Mask
                import base64
                import numpy as np

                def decode_capture(data, is_color=True):
                    """Decode image data from UnrealCV (handles both bytes and base64)."""
                    if data is None or data == "error":
                        return None

                    try:
                        # If data is already bytes (PNG format), decode directly
                        if isinstance(data, bytes):
                            nparr = np.frombuffer(data, np.uint8)
                            mode = (
                                cv2.IMREAD_COLOR if is_color else cv2.IMREAD_UNCHANGED
                            )
                            img = cv2.imdecode(nparr, mode)
                            return img

                        # If data is a string, check if it's an error or base64
                        if isinstance(data, str):
                            # Short strings are likely error messages
                            if len(data) < 100:
                                return None

                            # Try base64 decode for longer strings
                            img_bytes = base64.b64decode(data)
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            mode = (
                                cv2.IMREAD_COLOR if is_color else cv2.IMREAD_UNCHANGED
                            )
                            img = cv2.imdecode(nparr, mode)
                            return img

                        return None
                    except Exception as e:
                        return None

                # 1. Capture RGB
                img_data = self.unreal.client.request(f"vget /camera/{cam_id}/lit png")
                img = decode_capture(img_data, is_color=True)
                if img is not None:
                    rgb_path = view_dir / "rgb" / f"frame_{self.frame_idx:06d}.png"
                    cv2.imwrite(str(rgb_path), img)
                    if self.frame_idx == 0:  # Only log first frame
                        print(f"✓ [{view['id']}] Captured RGB {img.shape}")

                # 2. Capture Depth (use npy format)
                depth_data = self.unreal.client.request(
                    f"vget /camera/{cam_id}/depth npy"
                )
                # Check for error response - handle both bytes and string types
                is_error = False
                if depth_data is None:
                    is_error = True
                elif isinstance(depth_data, str):
                    is_error = depth_data == "error" or depth_data.startswith("error")
                elif isinstance(depth_data, bytes):
                    is_error = depth_data == b"error" or depth_data.startswith(b"error")

                if depth_data and not is_error:
                    try:
                        import base64
                        import numpy as np

                        # Depth returns base64-encoded npy data
                        depth_bytes = base64.b64decode(depth_data)
                        depth_array = np.load(
                            io.BytesIO(depth_bytes), allow_pickle=True
                        )
                        depth_path = (
                            view_dir / "depth" / f"frame_{self.frame_idx:06d}.npy"
                        )
                        np.save(str(depth_path), depth_array)
                        if self.frame_idx == 0:
                            print(
                                f"✓ [{view['id']}] Captured Depth {depth_array.shape}"
                            )
                    except Exception as e:
                        if self.frame_idx == 0:
                            print(f"✗ [{view['id']}] Depth decode error: {e}")
                elif self.frame_idx == 0:
                    print(f"✗ [{view['id']}] Depth error: {depth_data}")

                # 3. Capture Mask
                mask_data = self.unreal.client.request(
                    f"vget /camera/{cam_id}/object_mask png"
                )
                mask_img = decode_capture(mask_data, is_color=True)
                if mask_img is not None:
                    mask_path = view_dir / "mask" / f"frame_{self.frame_idx:06d}.png"
                    cv2.imwrite(str(mask_path), mask_img)

            except Exception as e:
                self.logger.warning(f"Failed to capture from view {view['id']}: {e}")

        self.frame_idx += 1
