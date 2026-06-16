# Plug Pose Estimation

Phase 1 verifies that the RealSense `.bag` files contain RGB-D streams and that the STL model loads correctly. The tagged and untagged recordings are separate sequences, so the tagged recording is used as a reference/calibration sequence rather than paired ground truth for the untagged recording.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy scipy pandas opencv-contrib-python pyrealsense2 open3d trimesh tqdm pyyaml
pip freeze > requirements.txt
```

## Phase 1

Inspect the RealSense bags:

```powershell
python scripts/inspect_bag.py data/raw/plug_with_tag_1280.bag --out outputs/phase1/bag_summary_with_tag.txt
python scripts/inspect_bag.py data/raw/plug_without_tag_1280.bag --out outputs/phase1/bag_summary_without_tag.txt
```

Extract aligned RGB-D sample frames:

```powershell
python scripts/extract_frames.py --bag data/raw/plug_with_tag_1280.bag --out data/extracted/with_tag --max_frames 100
python scripts/extract_frames.py --bag data/raw/plug_without_tag_1280.bag --out data/extracted/without_tag --max_frames 100
```

Inspect the STL:

```powershell
python scripts/view_stl.py data/raw/plug.STL --out outputs/phase1/stl_summary.txt
```

The STL mesh is loaded and its bounding box is inspected to determine scale. The STL coordinate frame is treated as the canonical plug frame.

## Phase 2

Detect the ArUco marker in extracted frames:

```powershell
python scripts/detect_aruco_debug.py --frames data/extracted/with_tag --out outputs/phase2/aruco_debug
```

Run the ArUco pose baseline:

```powershell
python scripts/run_aruco_pose.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --marker_length 0.011 ^
  --marker_id 32 ^
  --out_csv outputs/phase2/poses_with_tag_marker.csv ^
  --out_video outputs/phase2/with_tag_aruco_overlay.mp4
```

The first stage of the pipeline reads the RealSense `.bag` recordings using `pyrealsense2`, aligns depth to the color stream, extracts RGB-D frames, and records the color camera intrinsics. I also load the provided STL mesh and inspect its bounding box to determine the mesh scale and define a canonical plug coordinate frame.

The second stage implements an ArUco-based reference pose pipeline for the tagged sequence. The marker is detected in the color stream using the 5x5 ArUco dictionary with ID 32 and a side length of 11 mm. Using the color camera intrinsics, I estimate the 6-DoF pose of the marker relative to the camera for each frame where the marker is visible. The output is a per-frame pose trajectory and an overlay video showing the estimated coordinate frame.

Because the tagged and untagged recordings are separate sequences, I treat the tagged bag as a reference and calibration sequence rather than direct paired ground truth for the untagged bag. The ArUco pose gives `T_camera_marker`. Estimating the true plug pose additionally requires a fixed `T_marker_plug` transform between the marker and the canonical STL plug frame.

## Phase 2.5

Convert the marker pose into an approximate plug-frame pose using the provisional marker-to-plug transform in `config.yaml`:

```powershell
python scripts/run_tagged_plug_pose.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --marker_csv outputs/phase2/poses_with_tag_marker.csv ^
  --config config.yaml ^
  --out_csv outputs/phase2/poses_with_tag_plug.csv ^
  --out_video outputs/phase2/with_tag_plug_overlay.mp4
```

This computes:

```text
T_camera_plug = T_camera_marker @ T_marker_plug
```

The default `T_marker_plug` is an initial estimate based on the STL bounding-box length and the assumption that the marker is attached near one end of the plug. The plug is a long rectangular object; when it hangs naturally, the marker is on the lower end and the plug center should be above the marker.

Correction note: an earlier provisional config used `translation_m: [0.0282, 0.0, 0.0]`. That applied the half-length offset along the marker's horizontal X direction, placing the plug frame to the side of the marker in empty space. A later vertical offset of `translation_m: [0.0, -0.0282, 0.0]` used the full plug half-length from the marker center. Since ArUco pose starts at the marker center and the full 11 mm marker is pasted on the lower end of the plug, the current estimate subtracts half the marker length: `28.225 mm - 5.5 mm = 22.725 mm`. The current provisional translation is `translation_m: [0.0, -0.022725, 0.0]`.

Orientation note: the STL bounding box showed the long dimension left-right when `rotation_rpy_deg` was `[0.0, 0.0, 0.0]`. That means the STL long axis was still aligned to marker X. The plug hangs vertically through the marker, so the current provisional config uses `rotation_rpy_deg: [0.0, 0.0, -90.0]` to rotate the STL long axis onto marker -Y. The sign and rotation should still be checked visually against `outputs/phase2/with_tag_plug_overlay.mp4` and `outputs/phase2/with_tag_plug_bbox_overlay.mp4`.

Depth note: the marker pose origin is on the visible marker plane, while the STL bounding-box origin is inside the plug. Since marker +Z/blue points out toward the camera, the plug body lies behind the marker plane. The current provisional config shifts the plug center by half the STL thickness along marker -Z: `translation_m: [0.0, -0.020725, -0.007345]`.

To visualize the STL bounding box using the current plug-frame estimate:

```powershell
python scripts/run_tagged_plug_bbox_overlay.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --marker_csv outputs/phase2/poses_with_tag_marker.csv ^
  --config config.yaml ^
  --stl data/raw/plug.STL ^
  --out_video outputs/phase2/with_tag_plug_bbox_overlay.mp4
```

## Phase 3

Phase 3 starts the markerless RGB-D + STL baseline. The first validation step is run on the tagged recording while ignoring the tag for the depth/STL alignment. The ArUco-derived plug pose is used only as an initializer and reference for this validation pass.

Create a debug observed point cloud from one tagged frame:

```powershell
python scripts/debug_depth_pointcloud.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --config config.yaml ^
  --roi_key roi_tagged ^
  --frame_id 0 ^
  --out outputs/phase3/debug_observed_pcd/frame_000000.ply
```

Run one-frame markerless ICP:

```powershell
python scripts/run_markerless_icp.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --stl data/raw/plug.STL ^
  --config config.yaml ^
  --roi_key roi_tagged ^
  --init_csv outputs/phase2/poses_with_tag_plug.csv ^
  --reference_csv outputs/phase2/poses_with_tag_plug.csv ^
  --out_csv outputs/phase3/poses_with_tag_icp_frame0.csv ^
  --out_video outputs/phase3/with_tag_icp_frame0_overlay.mp4 ^
  --debug_dir outputs/phase3/debug_icp_frame0 ^
  --start_frame 0 ^
  --max_frames 1
```

The current baseline uses manual ROI cropping, depth filtering, pose-gated point filtering around the initialized STL volume, and point-to-point ICP. Rotation is locked for the first baseline because the visible depth is only a partial surface of a mostly rectangular object; unconstrained point-to-point ICP can rotate the model to a different but still low-RMSE alignment. This makes the first markerless result a translation-refinement baseline rather than a full unconstrained 6-DoF tracker.

Current frame-0 result:

```text
observed depth points: 4181 before pose gating
ICP fitness: 0.7925
ICP RMSE: 0.00346 m
translation error vs ArUco plug reference: 0.00639 m
rotation error vs ArUco plug reference: 0.0 deg, rotation locked to initializer
```

Short-window tracking validation on the tagged video:

```powershell
python scripts/run_markerless_icp.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --stl data/raw/plug.STL ^
  --config config.yaml ^
  --roi_key cluster_roi_tagged ^
  --init_csv outputs/phase2/poses_with_tag_plug.csv ^
  --init_first_only ^
  --reference_csv outputs/phase2/poses_with_tag_plug.csv ^
  --out_csv outputs/phase3/poses_with_tag_icp_track_127_170.csv ^
  --out_video outputs/phase3/with_tag_icp_track_127_170_overlay.mp4 ^
  --debug_dir outputs/phase3/debug_icp_track_127_170 ^
  --start_frame 127 ^
  --max_frames 44 ^
  --lock_rotation true
```

This initializes frame 127 from the ArUco-derived plug pose, then tracks frames 128-170 using the previous ICP pose. The ArUco poses are used only for comparison after frame 127. Current tagged-window result:

```text
tracked frames: 44
lost depth-gate frames: 0
frames with ArUco reference for comparison: 24
mean fitness: 0.7094
mean RMSE: 0.00333 m
mean translation error vs ArUco reference: 0.00768 m
mean rotation error vs ArUco reference: 16.41 deg
```

Cluster-assisted tracking uses the selected depth/color cluster as the current-frame measurement. After the first frame, cluster choice is biased toward continuity with the previous frame:

```text
previous selected mask -> dilate mask -> prefer clusters overlapping that dilated mask
```

Dilating a mask means expanding the white/selected pixels outward by a fixed pixel radius. This makes the previous-frame mask tolerant to normal motion between adjacent frames while still preserving the idea of "stay on the same object surface." The tracker also keeps the previous pose orientation after the first frame and uses the new cluster mainly to update translation and the observed ICP cloud.
