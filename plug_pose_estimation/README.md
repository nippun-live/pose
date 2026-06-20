# Plug Pose Estimation

This project estimates 6-DoF plug pose from Intel RealSense RGB-D recordings and an STL model.

The final pipeline is:

```text
RealSense .bag
-> aligned RGB-D frames
-> SAM2 plug mask
-> masked depth point cloud
-> centered STL point cloud
-> multiscale point-to-point ICP
-> fixed pose-frame correction learned from tagged ArUco validation
-> pose CSV and overlay video
```

The tagged recording is used for ArUco reference/calibration. The untagged recording is evaluated visually and diagnostically because it has no marker reference.

## Inputs

Expected local files:

```text
data/raw/plug_with_tag_1280.bag
data/raw/plug_without_tag_1280.bag
data/raw/plug.STL
```

Large input and output files are ignored by git.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy scipy pandas opencv-contrib-python pyrealsense2 open3d trimesh tqdm pyyaml torch
```

SAM2 is expected under:

```text
third_party/sam2
```

with a checkpoint such as:

```text
third_party/sam2/checkpoints/sam2.1_hiera_small.pt
```

## Final Scripts

The main script directory is intentionally small:

```text
scripts/
  inspect_inputs.py
  prepare_frames.py
  run_aruco_reference.py
  segment_sam2.py
  run_markerless_pose.py
  calibrate_pose_correction.py
  apply_pose_correction.py
  evaluate_pose.py
  render_pose_video.py
  refine_silhouette_fallback.py
```

Older exploratory scripts are preserved in:

```text
scripts/archive/
```

## 1. Inspect Inputs

```powershell
python scripts/inspect_inputs.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --bag data/raw/plug_without_tag_1280.bag ^
  --stl data/raw/plug.STL ^
  --out_json outputs/phase1/input_summary.json
```

This verifies bag streams, color intrinsics, depth scale, and STL bounding-box dimensions.

## 2. Prepare Frames for SAM2

Tagged video:

```powershell
python scripts/prepare_frames.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --out_dir outputs/phase4/run1/sam2_frames_with_tag
```

Untagged video:

```powershell
python scripts/prepare_frames.py ^
  --bag data/raw/plug_without_tag_1280.bag ^
  --out_dir outputs/phase4/run3_untagged_sam/sam2_frames_without_tag
```

This creates:

```text
rgb/
frame_map.json
intrinsics.json
streams.json
```

## 3. ArUco Reference on Tagged Video

```powershell
python scripts/run_aruco_reference.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --stl data/raw/plug.STL ^
  --config config.yaml ^
  --marker_length 0.011 ^
  --marker_id 32 ^
  --out_marker_csv outputs/phase2/poses_with_tag_marker.csv ^
  --out_plug_csv outputs/phase2/poses_with_tag_plug.csv ^
  --out_video outputs/phase2/with_tag_aruco_plug_reference.mp4
```

The marker pose is:

```text
T_camera_marker
```

The plug reference pose is:

```text
T_camera_plug = T_camera_marker @ T_marker_plug
```

`T_marker_plug` is configured in `config.yaml`. It encodes the fixed placement of the marker on the plug and defines the plug frame at the centered STL bounding-box frame.

## 4. Segment Plug with SAM2

Example tagged segmentation command:

```powershell
python scripts/segment_sam2.py ^
  --frames_dir outputs/phase4/run1/sam2_frames_with_tag/rgb ^
  --frame_map outputs/phase4/run1/sam2_frames_with_tag/frame_map.json ^
  --sam2_repo third_party/sam2 ^
  --checkpoint third_party/sam2/checkpoints/sam2.1_hiera_small.pt ^
  --click_frame 0 ^
  --click_x 691 ^
  --click_y 299 ^
  --prompt 0:636:119:0 ^
  --out_dir outputs/phase4/run1/segmentation_masks
```

SAM2 outputs a binary plug mask per frame and a `mask_map.json`. The mask is only a 2D object selector. The 3D pose still comes from masked depth and STL ICP.

## 5. Run Markerless Multiscale ICP

Tagged validation run:

```powershell
python scripts/run_markerless_pose.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --mask_map outputs/phase4/run1/segmentation_masks/mask_map.json ^
  --stl data/raw/plug.STL ^
  --config config.yaml ^
  --reference_csv outputs/phase2/poses_with_tag_plug.csv ^
  --out_dir outputs/phase4/run1/multiscale_p2p_icp
```

Untagged run:

```powershell
python scripts/run_markerless_pose.py ^
  --bag data/raw/plug_without_tag_1280.bag ^
  --mask_map outputs/phase4/run3_untagged_sam/segmentation_masks/mask_map.json ^
  --stl data/raw/plug.STL ^
  --config config.yaml ^
  --out_dir outputs/phase4/run4_untagged_icp/multiscale_p2p_icp
```

For each masked depth pixel:

```text
Z = raw_depth * depth_scale
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

The observed point cloud is aligned to a centered STL point cloud with multiscale point-to-point ICP.

## 6. Calibrate Fixed Pose-Frame Correction

The raw ICP pose tracks the plug but uses a different local frame convention from the ArUco plug frame. A fixed correction is learned on a random 60 percent of ArUco-visible tagged frames and validated on the held-out 40 percent.

```powershell
python scripts/calibrate_pose_correction.py ^
  --gt_csv outputs/phase2/poses_with_tag_plug.csv ^
  --pred_csv outputs/phase4/run1/multiscale_p2p_icp/poses_markerless.csv ^
  --split_mode random ^
  --train_fraction 0.6 ^
  --seed 42 ^
  --out_csv outputs/phase4/run1/multiscale_p2p_icp/poses_multiscale_p2p_corrected_random60_seed42.csv ^
  --out_summary_csv outputs/phase4/run1/multiscale_p2p_icp/calibration_summary_random60_seed42.csv ^
  --out_train_frames_csv outputs/phase4/run2/final_tag_videos/multiscale_p2p_random60_seed42_train_frames.csv ^
  --out_validation_frames_csv outputs/phase4/run2/final_tag_videos/multiscale_p2p_random60_seed42_validation_frames.csv
```

The correction is:

```text
C_i = inverse(T_camera_plug_icp_i) @ T_camera_plug_aruco_i
T_camera_plug_corrected_i = T_camera_plug_icp_i @ average(C_i)
```

Final tagged validation:

```text
translation error: 4.40 mm mean
rotation error: 11.98 deg mean
```

## 7. Apply Correction to Untagged Video

```powershell
python scripts/apply_pose_correction.py ^
  --poses outputs/phase4/run4_untagged_icp/multiscale_p2p_icp/poses_markerless.csv ^
  --correction_summary_csv outputs/phase4/run1/multiscale_p2p_icp/calibration_summary_random60_seed42.csv ^
  --out_csv outputs/phase4/run4_untagged_icp/multiscale_p2p_icp/poses_multiscale_p2p_corrected_random60_seed42.csv
```

## 8. Evaluate Tagged Results

```powershell
python scripts/evaluate_pose.py ^
  --gt_csv outputs/phase2/poses_with_tag_plug.csv ^
  --pred raw=outputs/phase4/run1/multiscale_p2p_icp/poses_markerless.csv ^
  --pred corrected=outputs/phase4/run1/multiscale_p2p_icp/poses_multiscale_p2p_corrected_random60_seed42.csv ^
  --out_summary_csv outputs/phase4/run1/evaluation_summary.csv ^
  --out_per_frame_csv outputs/phase4/run1/evaluation_per_frame.csv
```

## 9. Render Final Videos

Tagged raw versus ArUco:

```powershell
python scripts/render_pose_video.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --pose_a outputs/phase4/run1/multiscale_p2p_icp/poses_markerless.csv ^
  --pose_b outputs/phase2/poses_with_tag_plug.csv ^
  --label_a "raw multiscale P2P" ^
  --label_b "ArUco reference" ^
  --stl data/raw/plug.STL ^
  --out_video outputs/phase4/run2/final_tag_videos/multiscale_p2p_raw_vs_aruco_bbox_pose.mp4
```

Tagged corrected versus ArUco:

```powershell
python scripts/render_pose_video.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --pose_a outputs/phase4/run1/multiscale_p2p_icp/poses_multiscale_p2p_corrected_random60_seed42.csv ^
  --pose_b outputs/phase2/poses_with_tag_plug.csv ^
  --label_a "corrected multiscale P2P" ^
  --label_b "ArUco reference" ^
  --stl data/raw/plug.STL ^
  --out_video outputs/phase4/run2/final_tag_videos/multiscale_p2p_corrected_random60_vs_aruco_bbox_pose.mp4
```

Untagged corrected:

```powershell
python scripts/render_pose_video.py ^
  --bag data/raw/plug_without_tag_1280.bag ^
  --pose_a outputs/phase4/run4_untagged_icp/multiscale_p2p_icp/poses_multiscale_p2p_corrected_random60_seed42.csv ^
  --label_a "corrected markerless pose" ^
  --stl data/raw/plug.STL ^
  --out_video outputs/phase4/run4_untagged_icp/videos/multiscale_p2p_corrected_random60_bbox_pose.mp4
```

## Optional Silhouette Fallback

For short untagged spans where SAM masks were good but depth was incomplete, a bounded 2D silhouette refinement can be applied:

```powershell
python scripts/refine_silhouette_fallback.py ^
  --bag data/raw/plug_without_tag_1280.bag ^
  --poses outputs/phase4/run4_untagged_icp/multiscale_p2p_icp/poses_multiscale_p2p_corrected_random60_seed42.csv ^
  --mask_map outputs/phase4/run3_untagged_sam/segmentation_masks/mask_map.json ^
  --stl data/raw/plug.STL ^
  --frames 86-96 124-143 ^
  --out_csv outputs/phase4/run6_untagged_final_run4_source/poses/poses_corrected_random60_seed42_imagefallback.csv ^
  --out_dir outputs/phase4/run6_untagged_final_run4_source/image_fallback
```

This is an explicit fallback for bad-depth intervals, not the main pose estimator.
