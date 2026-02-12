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

    def __init__(
        self, unreal_client, output_dir, camera_configs=None, skip_depth=False
    ):
        """
        Args:
            unreal_client: UnrealCV API client
            output_dir: Root directory for saving recordings
            skip_depth: If True, skip depth capture (avoids hangs on some UnrealCV setups)
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
        self.skip_depth = skip_depth

        # Initialize logger early
        self.logger = logging.getLogger(__name__)

        # Legacy API support
        self.views = []  # For dynamically added views
        self._capture_resolution = (1280, 720)  # Default; updated by set_resolution
        self._default_fov = 90  # Used for intrinsics when FOV not available
        self._legacy_metadata_saved = False
        self._last_follow_state = {}  # {view_id: (x, y, z, pitch, yaw)} for smoothing
        self._follow_smoothing = (
            0.35  # 0=snap, 1=no movement; ~0.35 gives smooth tracking
        )
        self._camera_settle_delay = 0.02  # Per-view: wait after positioning before capture

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
        self._capture_resolution = (width, height)
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

    def _save_legacy_camera_metadata(self):
        """
        Save camera intrinsics for all legacy views.
        Called once on first frame capture.
        """
        if self._legacy_metadata_saved or not self.views:
            return
        width, height = self._capture_resolution
        fov = self._default_fov
        fx = fy = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        cx, cy = width / 2.0, height / 2.0
        metadata = {"cameras": []}
        for view in self.views:
            intrinsics = {
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "width": width,
                "height": height,
                "fov": fov,
            }
            cam_meta = {
                "id": view["id"],
                "type": view["type"],
                "intrinsics": intrinsics,
                "initial_position": view.get("location", [0, 0, 0]),
                "initial_rotation": view.get("rotation", [0, 0, 0]),
            }
            metadata["cameras"].append(cam_meta)
        meta_path = self.output_dir / "camera_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self._legacy_metadata_saved = True
        self.logger.info(f"Saved legacy camera metadata to {meta_path}")

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
        self._last_follow_state = {}
        self.logger.info("Recorder reset")

    # =================================================================
    # Legacy API for backward compatibility with test_pipeline.py
    # =================================================================

    def _spawn_virtual_camera(self):
        """
        Spawn a new virtual camera and return its ID.

        Requires UnrealCV v0.4.0+ with multi-camera support.
        The new camera is assigned the next sequential ID.
        """
        num_before = 0
        if hasattr(self.unreal, "get_camera_num"):
            num_before = self.unreal.get_camera_num()
        else:
            cameras_resp = self.unreal.client.request("vget /cameras")
            if cameras_resp and cameras_resp != "error":
                try:
                    cameras = json.loads(cameras_resp)
                    num_before = (
                        len(cameras) if isinstance(cameras, list) else int(cameras_resp)
                    )
                except (json.JSONDecodeError, ValueError):
                    num_before = 0

        self.unreal.client.request("vset /cameras/spawn")

        if hasattr(self.unreal, "get_camera_num"):
            cam_id = self.unreal.get_camera_num() - 1
        else:
            cameras_resp = self.unreal.client.request("vget /cameras")
            if cameras_resp and cameras_resp != "error":
                try:
                    cameras = json.loads(cameras_resp)
                    cam_id = (
                        len(cameras) - 1 if isinstance(cameras, list) else num_before
                    )
                except (json.JSONDecodeError, ValueError):
                    cam_id = num_before
            else:
                cam_id = num_before

        self.logger.debug(f"Spawned virtual camera, cam_id={cam_id}")
        return cam_id

    def add_static_view(self, view_id, location, rotation):
        """
        Add a static camera view (legacy API).

        Args:
            view_id: Unique identifier for this view
            location: [x, y, z] camera position
            rotation: [pitch, yaw, roll] camera rotation
        """
        cam_id = self._spawn_virtual_camera()

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
        cam_id = self._spawn_virtual_camera()

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
        cam_id = self._spawn_virtual_camera()

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
        Saves intrinsics (camera_metadata.json) on first frame and extrinsics per frame.
        """
        import cv2

        # Save camera intrinsics metadata once
        self._save_legacy_camera_metadata()

        frame_meta = {
            "frame_idx": self.frame_idx,
            "timestamp": time.time(),
            "cameras": [],  # List of view ids; full metadata in each view's metadata/ subdir
        }

        # Single pass: for each view, update position (if dynamic), brief settle, then capture
        for view in self.views:
            view_dir = self.output_dir / view["id"]
            cam_id = view["cam_id"]

            try:
                if view["type"] == "actor_pov":
                    # Get actor location and rotation for true POV (camera rotates with actor)
                    actor_id = view["actor_id"]
                    loc_str = self.unreal.client.request(
                        f"vget /object/{actor_id}/location"
                    )
                    rot_str = self.unreal.client.request(
                        f"vget /object/{actor_id}/rotation"
                    )

                    if loc_str and loc_str != "error":
                        try:
                            import math

                            actor_loc = [float(x) for x in loc_str.split()]
                            offset_loc = view["offset_loc"]  # [forward, right, up] in actor local frame
                            offset_rot = view["offset_rot"]
                            # Position: apply offset in actor's local frame so camera stays in front of face (avoids head clipping)
                            if rot_str and rot_str != "error":
                                actor_rot = [float(x) for x in rot_str.split()]
                                yaw_rad = math.radians(actor_rot[1])
                                fwd_x = math.cos(yaw_rad)
                                fwd_y = math.sin(yaw_rad)
                                loc = [
                                    actor_loc[0]
                                    + offset_loc[0] * fwd_x
                                    - offset_loc[1] * math.sin(yaw_rad),
                                    actor_loc[1]
                                    + offset_loc[0] * fwd_y
                                    + offset_loc[1] * math.cos(yaw_rad),
                                    actor_loc[2] + offset_loc[2],
                                ]
                                rot = [
                                    actor_rot[i] + offset_rot[i] for i in range(3)
                                ]
                            else:
                                loc = [actor_loc[i] + offset_loc[i] for i in range(3)]
                                rot = offset_rot
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
                            import math

                            actor_loc = [float(x) for x in loc_str.split()]
                            actor_rot = [float(x) for x in rot_str.split()]

                            # Target follow position (behind actor based on yaw)
                            yaw_rad = math.radians(actor_rot[1] + view["yaw"])
                            distance = view["distance"]
                            target_x = actor_loc[0] - distance * math.cos(yaw_rad)
                            target_y = actor_loc[1] - distance * math.sin(yaw_rad)
                            target_z = actor_loc[2] + 100  # Slight elevation
                            target_yaw = actor_rot[1] + view["yaw"]

                            # Smooth camera movement to avoid jumping when actor turns
                            view_id = view["id"]
                            t = self._follow_smoothing
                            if view_id in self._last_follow_state:
                                prev_x, prev_y, prev_z, prev_yaw = (
                                    self._last_follow_state[view_id]
                                )
                                cam_x = prev_x + t * (target_x - prev_x)
                                cam_y = prev_y + t * (target_y - prev_y)
                                cam_z = prev_z + t * (target_z - prev_z)
                                # Lerp yaw via shortest path
                                delta = (target_yaw - prev_yaw) % 360
                                if delta > 180:
                                    delta -= 360
                                cam_yaw = prev_yaw + t * delta
                            else:
                                cam_x, cam_y, cam_z = target_x, target_y, target_z
                                cam_yaw = target_yaw

                            self._last_follow_state[view_id] = (
                                cam_x,
                                cam_y,
                                cam_z,
                                cam_yaw,
                            )

                            self.unreal.client.request(
                                f"vset /camera/{cam_id}/location {cam_x} {cam_y} {cam_z}"
                            )
                            self.unreal.client.request(
                                f"vset /camera/{cam_id}/rotation {view['pitch']} {cam_yaw} 0"
                            )
                        except (ValueError, IndexError) as e:
                            if self.frame_idx == 0:
                                print(
                                    f"✗ [{view['id']}] Failed to parse actor data: {e}"
                                )
                    elif self.frame_idx == 0:
                        print(f"✗ [{view['id']}] Actor '{actor_id}' not found")

                # Static cameras don't need position updates (set during add_static_view)

                # Brief pause after positioning so Unreal applies transform before capture
                if view["type"] in ("actor_pov", "actor_follow"):
                    time.sleep(self._camera_settle_delay)

            except Exception as e:
                self.logger.warning(f"Failed to update camera for view {view['id']}: {e}")

            try:
                # Build frame metadata: intrinsics + extrinsics
                cam_frame_meta = {}
                width, height = self._capture_resolution
                fov = self._default_fov
                fx = fy = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
                cx, cy = width / 2.0, height / 2.0
                cam_frame_meta["intrinsics"] = {
                    "fx": float(fx),
                    "fy": float(fy),
                    "cx": float(cx),
                    "cy": float(cy),
                    "width": width,
                    "height": height,
                    "fov": fov,
                }
                loc_str = self.unreal.client.request(f"vget /camera/{cam_id}/location")
                rot_str = self.unreal.client.request(f"vget /camera/{cam_id}/rotation")
                if loc_str and loc_str != "error" and rot_str and rot_str != "error":
                    try:
                        position = [float(x) for x in loc_str.split()]
                        rotation = [float(x) for x in rot_str.split()]
                        cam_frame_meta["extrinsics"] = {
                            "position": position,
                            "rotation": rotation,
                        }
                    except (ValueError, IndexError):
                        pass

                # Capture Modalities: RGB, Depth, Mask
                import base64

                def decode_capture(data, is_color=True):
                    """Decode image data from UnrealCV (handles both bytes and base64)."""
                    if data is None or data == "error":
                        return None
                    try:
                        if isinstance(data, bytes):
                            nparr = np.frombuffer(data, np.uint8)
                            mode = cv2.IMREAD_COLOR if is_color else cv2.IMREAD_UNCHANGED
                            img = cv2.imdecode(nparr, mode)
                            return img
                        if isinstance(data, str):
                            if len(data) < 100:
                                return None
                            img_bytes = base64.b64decode(data)
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            mode = cv2.IMREAD_COLOR if is_color else cv2.IMREAD_UNCHANGED
                            img = cv2.imdecode(nparr, mode)
                            return img
                        return None
                    except Exception:
                        return None

                # 1. Capture RGB
                img_data = self.unreal.client.request(f"vget /camera/{cam_id}/lit png")
                img = decode_capture(img_data, is_color=True)
                if img is not None:
                    rgb_path = view_dir / "rgb" / f"frame_{self.frame_idx:06d}.png"
                    cv2.imwrite(str(rgb_path), img)
                    cam_frame_meta["rgb"] = str(rgb_path.relative_to(self.output_dir))

                # 2. Capture Depth (use npy format)
                if self.skip_depth:
                    depth_data = None
                else:
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

                        # UnrealCV returns raw float32 bytes (not .npy pickle)
                        if isinstance(depth_data, str):
                            depth_bytes = base64.b64decode(depth_data)
                        else:
                            depth_bytes = (
                                bytes(depth_data)
                                if hasattr(depth_data, "__len__")
                                else depth_data
                            )
                        depth_flat = np.frombuffer(depth_bytes, dtype=np.float32)
                        n_total = len(depth_flat)
                        h_exp, w_exp = (
                            self._capture_resolution[1],
                            self._capture_resolution[0],
                        )
                        n_exp = h_exp * w_exp
                        if n_total >= n_exp:
                            depth_array = depth_flat[-n_exp:].reshape(h_exp, w_exp, 1)
                        else:
                            # Virtual cameras may use env default (e.g. 160x160); infer dims
                            n_px = int(np.sqrt(n_total)) ** 2
                            side = int(np.sqrt(n_px))
                            depth_array = depth_flat[-n_px:].reshape(side, side, 1)
                        depth_path = (
                            view_dir / "depth" / f"frame_{self.frame_idx:06d}.npy"
                        )
                        np.save(str(depth_path), depth_array)
                        cam_frame_meta["depth"] = str(
                            depth_path.relative_to(self.output_dir)
                        )
                    except Exception:
                        pass

                # 3. Capture Mask
                mask_data = self.unreal.client.request(
                    f"vget /camera/{cam_id}/object_mask png"
                )
                mask_img = decode_capture(mask_data, is_color=True)
                if mask_img is not None:
                    mask_path = view_dir / "mask" / f"frame_{self.frame_idx:06d}.png"
                    cv2.imwrite(str(mask_path), mask_img)
                    cam_frame_meta["mask"] = str(mask_path.relative_to(self.output_dir))

                # Save per-camera metadata (intrinsics, extrinsics, paths) to view's metadata subdir
                cam_meta_dir = view_dir / "metadata"
                cam_meta_dir.mkdir(parents=True, exist_ok=True)
                cam_meta_path = cam_meta_dir / f"frame_{self.frame_idx:06d}.json"
                cam_frame_meta["frame_idx"] = self.frame_idx
                cam_frame_meta["timestamp"] = frame_meta["timestamp"]
                with open(cam_meta_path, "w") as f:
                    json.dump(cam_frame_meta, f, indent=2)

                frame_meta["cameras"].append(view["id"])

            except Exception as e:
                self.logger.warning(f"Failed to capture from view {view['id']}: {e}")

        # Save scene-level frame index (frame_idx, timestamp, camera list)
        frame_meta_path = (
            self.output_dir / "scene_metadata" / f"frame_{self.frame_idx:06d}.json"
        )
        frame_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(frame_meta_path, "w") as f:
            json.dump(frame_meta, f, indent=2)

        self.frame_idx += 1
