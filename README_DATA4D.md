# UnrealZoo 4D Data Pipeline

### Code Structure

- `unrealzoo/`
  - `unrealzoo/core/recorder.py` (multi-view capture + pose/joint/mesh recording)
  - `unrealzoo/core/director.py` (time evolution, events, background nav motion)
  - `unrealzoo/core/pipeline.py` (orchestration wrapper)
  - `unrealzoo/utils/ground_detection.py` (safe spawn generation/loading helpers)
- Top-level scripts:
  - `test_pipeline.py` (main practical recording loop used for actor+camera sessions)
  - `demo_4d_pipeline.py` (config-driven demo entry point)
  - `generate_safe_spawns.py` (build `safe_spawns.json`)
  - `visualize_poses.py` (joint/mesh visualization from recorded pose files)
  - `replay_data.py` (RGB/mask/depth replay utility)
  - `safe_spawns.json` (generated/curated spawn database)

---

## 2) Pipeline overview

At a high level:

1. Launch environment and connect UnrealCV.
2. Spawn actors at safe points.
3. Register background motion via `Director`.
4. Configure cameras and pose capture via `MultiViewRecorder`.
5. For each frame:
   - step simulation (`director.step(dt)`)
   - resume game briefly (`vset /action/game/resume`)
   - pause game (`vset /action/game/pause`)
   - capture synchronized multi-view data (`recorder.capture_frame()`)

Core modules:

- `Director`: controls temporal behavior (scheduled events + NavMesh random motion).
- `MultiViewRecorder`: saves RGB/depth/mask (+ optional flow), per-camera metadata, and actor pose payloads.
- `Pipeline`: optional config-driven orchestration wrapper.

---

## 3) Run the pipeline

### A) Generate safe spawns (once per map or when needed)

```bash
python generate_safe_spawns.py --env UnrealAgent-Greek_Island-ContinuousColor-v0
```

### B) Run the practical recorder loop

```bash
python test_pipeline.py
```

This creates a timestamped session under `demo_output_60fps/<YYYYMMDD_HHMMSS>/`.

### C) Optional utilities

```bash
python replay_data.py --dir demo_output_60fps
python visualize_poses.py --dir demo_output_60fps
```

---

## 4) Output structure

Typical session layout:

```text
demo_output_60fps/<session>/
  camera_metadata.json
  scene_metadata/frame_000000.json
  poses/frame_000000.json
  poses/canonical_mesh/<actor>.npy
  <view_id>/rgb/frame_000000.png
  <view_id>/mask/frame_000000.png
  <view_id>/depth/frame_000000.npy (or .png depending on backend path)
  <view_id>/metadata/frame_000000.json
```

Notes:

- `scene_metadata/*`: frame index, timestamps, and camera-level capture references.
- `poses/*`: actor transforms, optional joints, optional canonical mesh metadata.
- `canonical_mesh/*.npy`: one cached "reference mesh" per actor when enabled.

---

## 5) Joint data status (important)

- Recorder supports joint discovery/queries and diagnostics in `unrealzoo/core/recorder.py`.
- In current UEZoo builds, bone/joint `vget` endpoints may be unavailable. In that case:
  - transforms still record correctly,
  - joint fields may be empty or fail if strict mode is enabled.

Recommended fallback:

- Use this pipeline to bake actor state trajectories now (transforms, camera data).
- Add Unreal plugin-side extension later if true per-joint transforms are required online.

---

## 6) Known implementation caveat

- `unrealzoo/core/recorder.py` currently contains two `capture_frame` definitions; the later legacy-style method is the active one in practice.
- If you add a new mode (for example, bake-only/no-render), implement it in the active capture path.
