# Plug Pose Estimation Experiment Log

This document records the engineering path used to get from raw RealSense recordings to the current markerless RGB-D plus STL pose baseline. It is written as a repeatable experiment log, not just a final-method summary. The assignment goal is to estimate plug pose and also show how the problem was approached, what failed, what worked, and what should be tried next.

## Problem Setup

Input data:

```text
data/raw/plug_with_tag_1280.bag
data/raw/plug_without_tag_1280.bag
data/raw/plug.STL
```

The tagged recording is used as a reference/calibration sequence. The untagged recording is the eventual markerless target. Since these are separate recordings, the tagged video is not direct ground truth for the untagged video.

Camera and STL facts found during inspection:

```text
Camera: Intel RealSense D405
Resolution: 1280 x 720
FPS: 15
Color stream: RGB8
Depth stream: Z16
Tagged bag: 355 frames, about 23.6 seconds
Untagged bag: 276 frames, about 18.3 seconds
STL bbox extent: 56.45 mm x 11.85 mm x 14.69 mm
```

The STL is treated as already being in meters. The plug frame is defined at the STL bounding-box center.

## Phase 1: Data Inspection

Scripts:

```text
scripts/inspect_bag.py
scripts/extract_frames.py
scripts/view_stl.py
```

What was verified:

- Both `.bag` files open through `pyrealsense2`.
- Both contain aligned color and depth streams.
- RGB frames, raw depth `.npy`, depth visualization `.png`, and camera intrinsics can be saved.
- The STL loads in Open3D/trimesh and its metric scale is plausible.

Important output files:

```text
outputs/phase1/bag_summary_with_tag.txt
outputs/phase1/bag_summary_without_tag.txt
outputs/phase1/stl_summary.txt
data/extracted/with_tag/intrinsics.json
data/extracted/without_tag/intrinsics.json
```

This phase was important because all later pose work depends on correct intrinsics, aligned depth, and correct STL scale.

## Phase 2: ArUco Reference Pose

Scripts:

```text
scripts/detect_aruco_debug.py
scripts/run_aruco_pose.py
scripts/run_tagged_plug_pose.py
scripts/run_tagged_plug_bbox_overlay.py
```

Library used:

```text
OpenCV contrib ArUco module: cv2.aruco
Dictionary: cv2.aruco.DICT_5X5_100
Marker ID: 32
Marker side length: 0.011 m
```

The first ArUco output is:

```text
T_camera_marker
```

That is the pose of the marker sticker, not the pose of the plug body. To express a plug pose, a fixed transform is needed:

```text
T_camera_plug = T_camera_marker @ T_marker_plug
```

The current manually tuned marker-to-plug transform is in `config.yaml`:

```yaml
marker_to_plug:
  translation_m: [0.0, -0.020725, -0.007345]
  rotation_rpy_deg: [0.0, 0.0, -90.0]
```

Why this transform was needed:

- The marker origin is the center of the sticker.
- The marker is stuck on the lower end of the plug when the plug hangs vertically.
- The plug center is above the marker center, not sideways in air.
- The plug is a 3D object, so the plug center is behind the marker plane by about half the plug thickness.
- The STL long axis needed a `-90 deg` rotation to make the bbox run along the plug length.

Important output files:

```text
outputs/phase2/poses_with_tag_marker.csv
outputs/phase2/poses_with_tag_plug.csv
outputs/phase2/with_tag_aruco_overlay.mp4
outputs/phase2/with_tag_plug_bbox_overlay.mp4
```

Example ArUco detection debug:

![ArUco debug frame](outputs/phase2/aruco_debug/aruco_000000.png)

Result:

```text
ArUco marker detected in 242 / 355 tagged frames, about 68 percent.
```

## Why Marker Pose Was Not Enough

A recurring issue was that a marker pose can look correct while the plug-frame bbox is wrong. The marker is a plane attached to one end of a 3D object. The plug pose must represent the STL coordinate frame, not the sticker plane.

The key correction was to move the plug origin:

```text
from marker plane center
to STL bounding-box center
```

This included:

- Long-axis offset from marker end toward plug center.
- Depth/thickness offset from visible marker face into the object volume.
- Rotation alignment between marker axes and STL axes.

This matters for Phase 3 because ICP estimates the STL pose. If the ArUco reference uses a marker-surface frame while ICP uses the STL-center frame, the comparison will show a constant offset even when both methods are internally correct.

## Phase 3 Goal

The markerless target is:

```text
RGB-D frame + STL model -> T_camera_plug
```

The classical baseline is:

```text
RGB-D frame
-> segment visible plug region
-> backproject depth pixels to 3D point cloud
-> sample centered STL as model point cloud
-> initialize pose
-> run ICP
-> save pose CSV and bbox/mask overlays
```

## Phase 3 Early Attempt: Manual ROI

The first markerless attempt used a manually defined ROI and depth range. This was useful for proving the pipeline but not robust enough.

Main issue:

```text
In early frames, plug depth is too close to background depth.
```

This caused the observed point cloud to include background surfaces and made the point cloud not look like the plug.

Example early segmentation diagnostics:

![Frame 0 segmentation RGB](outputs/phase3/verify_segmentation_frame0/frame_000000_segmentation_rgb.png)

![Frame 0 segmentation depth](outputs/phase3/verify_segmentation_frame0/frame_000000_segmentation_depth.png)

Lesson:

```text
Do not judge the pipeline only from frame 0. Pick frames where the plug is lifted and depth separation is better.
```

## Choosing Frame 127

Frame review sheets were generated to inspect candidate frames visually:

```text
outputs/phase3/frame_review_with_tag/sheets/
outputs/phase3/frame_review_without_tag/sheets/
```

Example review sheet:

![Frame review sheet](outputs/phase3/frame_review_with_tag/sheets/sheet_004.png)

Frame 127 was selected because the plug is visible, lifted, and has better separation than the first frame.

## Segmentation Method That Worked Best

The best segmentation baseline used:

```text
1. Broad ROI around the plug area.
2. Depth range filtering.
3. Simple HSV color filter to suppress saturated/colored clutter.
4. Backprojection of valid depth pixels to 3D.
5. DBSCAN clustering in 3D.
6. Cluster selection using geometry and tracking priors.
```

The selected cluster for frame 127 was manually identified as cluster 6 after inspecting the cluster overlays. This was not chosen by a learned model. The script generated labeled clusters, and visual inspection showed that cluster 6 corresponded to the visible plug face.

Frame 127 selected cluster:

![Frame 127 selected cluster RGB](outputs/phase3/depth_clusters_frame127_select6/frame_000127_selected_cluster_rgb.png)

![Frame 127 selected cluster mask](outputs/phase3/depth_clusters_frame127_select6/frame_000127_selected_cluster_mask.png)

Important limitation:

```text
The depth sensor sees mostly the visible face of the plug, not the whole plug volume.
```

The selected point cloud therefore looks like a thin surface patch with some holes. This is expected with a single RGB-D view.

To view the PLY:

```powershell
python -c "import open3d as o3d; p=o3d.io.read_point_cloud('outputs/phase3/depth_clusters_frame127_select6/frame_000127_selected_cluster.ply'); o3d.visualization.draw_geometries([p])"
```

PLY was used because it stores point coordinates and optional colors in a standard format that Open3D can read/write directly. A `.pcd` file would also work; PLY was just the format used by these scripts.

## STL Centering Detail

The STL vertices are not assumed to be centered at the desired plug frame. The desired plug frame is the STL bounding-box center. Before ICP, the sampled STL point cloud is centered around the STL bbox center.

This matters because:

```text
ICP pose should represent T_camera_plug,
where plug origin = STL bbox center.
```

If the STL point cloud were not centered, the pose CSV and bbox overlays would have a systematic offset.

Relevant code:

```text
src/plug_pose/stl_utils.py
src/plug_pose/project_mesh.py
```

## Pose Initialization Experiments

Several initialization strategies were tested.

### ArUco Initialization

For the tagged validation sequence, ArUco-derived plug pose can initialize ICP. This is allowed for validation because the marker is ignored during markerless estimation, but it is not usable for the untagged video.

Finding:

```text
ArUco initialization is useful for debugging, but it hides the harder problem of initializing pose without a marker.
```

### Cluster/PCA Initialization

A no-ArUco initialization was then tested on frame 127:

```text
1. Segment the plug region by depth/color clustering.
2. Estimate long direction from PCA of the selected cluster.
3. Estimate visible face normal.
4. Shift from visible face into the plug by half STL thickness.
5. Use this as the initial STL pose.
```

This gave a good enough first-frame pose without using ArUco.

Frame 127 bbox before/after from cluster initialization:

![Cluster init locked](outputs/phase3/cluster_init_bbox_before_after_frame127_locked.png)

![Cluster init unlocked](outputs/phase3/cluster_init_bbox_before_after_frame127_unlocked.png)

Finding:

```text
Cluster/PCA initialization is good enough for frame 127, but orientation from PCA is unstable across frames because the depth surface is partial and sometimes contaminated.
```

## ICP Experiments

ICP was run between:

```text
source: sampled centered STL point cloud
target: observed segmented RGB-D point cloud
```

Open3D point-to-point ICP was used:

```text
open3d.pipelines.registration.registration_icp
```

The point-to-point version was chosen first because observed normals are unreliable on sparse, hole-filled depth patches.

### Locked Rotation ICP

Locked rotation means:

```text
ICP is allowed to update translation,
but final rotation is reset to the initialization rotation.
```

This was tested because fully unlocked ICP often rotated the rectangular STL by about 80 to 90 degrees while still producing a good numerical surface fit.

Frame 127 locked/unlocked comparison:

![Frame 127 locked ICP](outputs/phase3/verify_icp_frame127_locked.png)

![Frame 127 unlocked ICP](outputs/phase3/verify_icp_frame127_unlocked.png)

Finding:

```text
Locked rotation preserves coordinate-frame consistency, but it cannot follow real orientation changes later in the video.
```

### Previous-Pose Tracking

After one frame works, each next frame can use the previous pose as the initialization. This is standard tracking.

Finding:

```text
Previous-pose tracking works while motion is small,
but if the plug rotates quickly, the projected bbox gate becomes stale.
```

This caused later-frame drift.

## Tracking Variants Tried

All variants below were tested on tagged frames 127 to 170 so the result could be compared against the ArUco-derived plug pose.

## What Actually Fixed Tracking and Bbox Placement

The final tracking behavior came from several separate fixes. These are easy to confuse, so this section states them explicitly.

First, segmentation was improved independently of pose:

```text
broad ROI
-> depth range filter
-> low-saturation color filter
-> 3D DBSCAN clustering
-> cluster size/shape checks
```

This produced a usable visible plug-surface mask. It did not produce a full 6-DoF pose by itself.

Second, the raw previous mask was demoted. V4 showed that previous-mask overlap can keep the plug in the mask, but once the mask touches the hand/wire/connector, that contamination gets propagated. V5 fixed this by using the projected STL bbox from the previous pose as the stronger spatial gate.

```text
V4: previous raw mask has too much authority
V5: previous STL/bbox pose gate has primary authority
```

This is what fixed the masks around frames 165 onward.

Third, orientation handling was separated from translation handling. In V5 and V6 the tracker used:

```text
current depth cluster -> update translation
previous pose -> keep orientation
locked ICP -> refine translation only
```

This made early frames stable, but it meant that if the plug rotated in the hand, the orange initialization bbox could still have stale orientation. The mask could be good while the bbox orientation was wrong.

Fourth, V7 allowed full ICP rotation:

```text
current depth cluster -> update translation
previous full-ICP pose -> initialize orientation
unconstrained ICP -> update translation and rotation
```

This is why V7 made both orange and green boxes look more aligned. The orange box in V7 inherits the previous frame's unconstrained ICP orientation, so it is no longer stuck at the old locked orientation.

The important lesson is:

```text
good mask != good bbox orientation
```

The mask fixes were mostly solved by V5. The bbox orientation improved only when V7 allowed the pose itself to rotate.

### V3: Cluster Translation + Previous Orientation

Idea:

```text
Use the fresh cluster to update where the plug is.
Keep orientation from previous pose.
Run locked ICP for small translation refinement.
```

Reason:

```text
PCA orientation from a partial depth surface was unstable.
```

This helped early frames, but still drifted when cluster selection changed.

### V4: Previous-Mask Overlap

Idea:

```text
Prefer clusters that overlap the previous selected mask.
Dilate the previous mask so small motion still overlaps.
```

Dilate means expanding the selected white mask outward by a number of pixels. This makes the previous mask more forgiving when the object moves slightly between frames.

Finding:

```text
Previous-mask overlap kept selecting something containing the plug,
but after about frame 165 it also pulled in hand/wire/connector pixels.
```

V4 frame 165 mask:

![V4 frame 165 mask](outputs/phase3/cluster_assisted_track_127_170_v4_prev_mask/frames/masks/frame_000165_mask_overlay.png)

This was a useful failure: the mask was not fragmented, but it became contaminated.

### V5: STL Pose-Gated Mask

Idea:

```text
Use previous pose / projected STL bbox as the main spatial gate.
Use previous mask only as a weak score.
Reject clusters much larger than the STL dimensions.
```

This fixed the V4 contamination problem better than relying on previous raw masks.

V5 frame 165 pose gate and mask:

![V5 frame 165 pose gate](outputs/phase3/cluster_assisted_track_127_170_v5_pose_gate/frames/pose_gate/frame_000165_pose_gate.png)

![V5 frame 165 mask](outputs/phase3/cluster_assisted_track_127_170_v5_pose_gate/frames/masks/frame_000165_mask_overlay.png)

Finding:

```text
The mask became much cleaner and stopped grabbing the large hand region.
```

Remaining issue:

```text
The bbox could still become wrong because pose tracking, especially orientation, was not robust.
```

Important detail about the orange bbox:

```text
In V5, orange is pre-ICP, but it is not fresh depth/PCA orientation after frame 127.
```

The orientation mode was:

```yaml
orientation_mode: previous_after_first
```

Therefore:

```text
Frame 127:
  orange orientation = cluster/PCA orientation from the selected depth surface

Frames 128+:
  orange translation = current depth cluster
  orange orientation = previous pose orientation
```

Because V5 used locked rotation, the previous pose orientation stayed stale when the plug rotated. That is why the orange bbox could be incorrectly oriented even though the mask was good.

### V6: ICP Acceptance Gate

Idea:

```text
Evaluate the initialized pose and the ICP pose.
Accept ICP only if it improves the fit and does not jump too far.
```

This was motivated by visual inspection: in many early frames, the initialized bbox looked better than the post-ICP bbox.

Finding:

```text
The gate rejected ICP in all 44 tested frames.
```

That showed raw ICP often wanted to move the model by 8 to 13 mm from initialization. This validated the suspicion that ICP was not always trustworthy.

But V6 was not the best result because rejecting ICP everywhere reduced visible alignment.

### V7: Unconstrained ICP

The most recent test removed the rotation lock and disabled the ICP acceptance gate:

```text
no --lock_rotation
--disable_icp_acceptance
```

Command:

```powershell
python scripts/run_cluster_assisted_tracker.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --stl data/raw/plug.STL ^
  --config config.yaml ^
  --roi_key cluster_roi_tagged ^
  --reference_csv outputs/phase2/poses_with_tag_plug.csv ^
  --out_dir outputs/phase3/cluster_assisted_track_127_170_v7_full_rotation ^
  --start_frame 127 ^
  --max_frames 44 ^
  --disable_icp_acceptance
```

V7 frame examples:

![V7 frame 127 bbox](outputs/phase3/cluster_assisted_track_127_170_v7_full_rotation/frames/bbox/frame_000127_bbox_overlay.png)

![V7 frame 170 bbox](outputs/phase3/cluster_assisted_track_127_170_v7_full_rotation/frames/bbox/frame_000170_bbox_overlay.png)

Finding:

```text
Unconstrained ICP gives the best visual bbox placement and best surface fit,
but it disagrees strongly with ArUco orientation.
```

Important detail about why V7 orange also improved:

```text
V7 still uses previous orientation after frame 127,
but that previous orientation now comes from the previous frame's full ICP result.
```

So:

```text
Frame 127:
  orange = cluster/PCA initialization
  green = full ICP result

Frames 128+:
  orange = current cluster translation + previous frame's full-ICP orientation
  green = current full ICP result
```

This explains why both orange and green appear better oriented in V7 than in V5. V7 did not magically make the orange box a fresh per-frame PCA estimate; it made the previous orientation better by allowing the previous green ICP pose to rotate.

This is likely because ICP is fitting a partial visible face of a rectangular object. The model can rotate around an ambiguous direction and still fit the visible surface well.

### V8: Bounded-Rotation ICP

After deciding to treat the ArUco-derived plug pose as ground truth, a bounded-rotation variant was added. The purpose was to keep V7's good translation/surface placement while reducing the large 80 to 90 degree rotation disagreement.

Implementation:

```text
1. Run full ICP.
2. Keep the full ICP translation.
3. Measure the rotation step from the pre-ICP pose to the full ICP pose.
4. If the step exceeds a limit, clamp the rotation update to that limit.
5. Save whether rotation was bounded in the output CSV.
```

The tracker options are:

```text
--bounded_rotation_deg <degrees>
--rotation_gate_mode clamp
```

Example command:

```powershell
python scripts/run_cluster_assisted_tracker.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --stl data/raw/plug.STL ^
  --config config.yaml ^
  --roi_key cluster_roi_tagged ^
  --reference_csv outputs/phase2/poses_with_tag_plug.csv ^
  --out_dir outputs/phase3/cluster_assisted_track_127_170_v8_rot30 ^
  --start_frame 127 ^
  --max_frames 44 ^
  --disable_icp_acceptance ^
  --bounded_rotation_deg 30 ^
  --rotation_gate_mode clamp
```

Rotation caps tested:

```text
15 deg, 30 deg, 45 deg, 60 deg
```

Finding:

```text
Bounded rotation did not solve the ArUco rotation error.
```

Reason:

```text
The bad full-ICP rotation is not a one-frame spike.
It is a persistent surface-fit direction.
The bounded update only slows the drift, but over several frames the
orientation still accumulates toward the same wrong ICP orientation.
```

Therefore, if the evaluation target is strict 6-DoF agreement with the ArUco plug pose, V5 is still better than V7/V8 for orientation.

## Quantitative Comparison on Tagged Frames 127-170

The comparison uses only frames where the ArUco-derived plug pose is available. This reference is approximate because it depends on the manually tuned `T_marker_plug`.

| Variant | Description | Tracked Frames | Reference Frames | Mean Fitness | Mean RMSE | Mean Translation Error | Mean Rotation Error |
|---|---|---:|---:|---:|---:|---:|---:|
| V5 | pose-gated mask + locked rotation | 44 | 24 | 0.602 | 3.469 mm | 8.08 mm | 16.45 deg |
| V6 | pose-gated mask + reject ICP | 44 | 24 | 0.488 | 3.548 mm | 10.50 mm | 16.45 deg |
| V7 | pose-gated mask + unconstrained ICP | 44 | 24 | 0.934 | 3.292 mm | 7.85 mm | 80.83 deg |
| V8-15 | bounded rotation, 15 deg/frame | 44 | 24 | 0.923 | 3.318 mm | 7.90 mm | 76.38 deg |
| V8-30 | bounded rotation, 30 deg/frame | 44 | 24 | 0.927 | 3.309 mm | 7.90 mm | 78.56 deg |
| V8-45 | bounded rotation, 45 deg/frame | 44 | 24 | 0.929 | 3.300 mm | 7.89 mm | 78.49 deg |
| V8-60 | bounded rotation, 60 deg/frame | 44 | 24 | 0.930 | 3.295 mm | 7.96 mm | 79.22 deg |

Interpretation:

- V7 is best for visible bbox placement and surface fit.
- V7 is slightly best in plug-center translation error.
- V5/V6 agree more with the ArUco orientation, but visually the boxes are less satisfying.
- V8 did not improve the ArUco rotation error enough to replace V5 for strict 6-DoF evaluation.
- The ArUco reference is useful, but not perfect ground truth because `T_marker_plug` was manually tuned.
- The large V7 rotation error likely reflects coordinate-frame ambiguity under partial-depth ICP, not necessarily total failure of bbox placement.

The comparison table was generated with:

```powershell
python scripts/evaluate_markerless_vs_aruco.py ^
  --gt_csv outputs/phase2/poses_with_tag_plug.csv ^
  --pred v5_locked=outputs/phase3/cluster_assisted_track_127_170_v5_pose_gate/poses_cluster_assisted.csv ^
  --pred v6_init_only=outputs/phase3/cluster_assisted_track_127_170_v6_icp_acceptance/poses_cluster_assisted.csv ^
  --pred v7_full=outputs/phase3/cluster_assisted_track_127_170_v7_full_rotation/poses_cluster_assisted.csv ^
  --pred v8_rot15=outputs/phase3/cluster_assisted_track_127_170_v8_rot15/poses_cluster_assisted.csv ^
  --pred v8_rot30=outputs/phase3/cluster_assisted_track_127_170_v8_rot30/poses_cluster_assisted.csv ^
  --pred v8_rot45=outputs/phase3/cluster_assisted_track_127_170_v8_rot45/poses_cluster_assisted.csv ^
  --pred v8_rot60=outputs/phase3/cluster_assisted_track_127_170_v8_rot60/poses_cluster_assisted.csv ^
  --out_summary_csv outputs/phase3/aruco_eval_summary.csv ^
  --out_per_frame_csv outputs/phase3/aruco_eval_per_frame.csv
```

## Full Tagged-Video Check of the V7 Correction

The first V7 correction result was based only on frames 127 to 170. To test whether this was a stable systematic offset over the longer tagged sequence, V7 was rerun from frame 127 to the end of the tagged bag:

```powershell
python scripts/run_cluster_assisted_tracker.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --stl data/raw/plug.STL ^
  --config config.yaml ^
  --roi_key cluster_roi_tagged ^
  --reference_csv outputs/phase2/poses_with_tag_plug.csv ^
  --out_dir outputs/phase3/cluster_assisted_track_127_end_v7_full_rotation ^
  --start_frame 127 ^
  --max_frames 228 ^
  --disable_icp_acceptance
```

This produced 228 markerless tracked frames, with 139 frames overlapping ArUco detections.

A new script was added to fit and apply a fixed local correction:

```text
scripts/calibrate_pose_correction.py
```

The correction trained on the original reliable window, frames 127 to 170, was:

```text
rotation euler xyz = [-80.532 deg, -1.450 deg, 0.251 deg]
translation = [1.65 mm, -6.47 mm, -0.66 mm]
```

Command:

```powershell
python scripts/calibrate_pose_correction.py ^
  --gt_csv outputs/phase2/poses_with_tag_plug.csv ^
  --pred_csv outputs/phase3/cluster_assisted_track_127_end_v7_full_rotation/poses_cluster_assisted.csv ^
  --train_start 127 ^
  --train_end 170 ^
  --out_csv outputs/phase3/cluster_assisted_track_127_end_v7_corrected_from_127_170.csv ^
  --out_summary_csv outputs/phase3/v7_full_correction_from_127_170_summary.csv ^
  --eval_range early:127-170 ^
  --eval_range mid1:171-197 ^
  --eval_range mid2:198-221 ^
  --eval_range mid3:222-260 ^
  --eval_range late:261-354 ^
  --eval_range all:127-354
```

Results using the frames 127 to 170 correction:

| Range | Raw Translation | Raw Rotation | Corrected Translation | Corrected Rotation |
|---|---:|---:|---:|---:|
| 127-170 | 7.88 mm | 80.55 deg | 3.36 mm | 4.54 deg |
| 171-197 | 9.17 mm | 98.43 deg | 5.12 mm | 18.90 deg |
| 198-221 | 27.67 mm | 86.51 deg | 28.01 mm | 11.82 deg |
| 222-260 | 9.55 mm | 88.70 deg | 4.00 mm | 9.30 deg |
| 261-354 | 70.74 mm | 31.57 deg | 68.66 mm | 95.16 deg |
| 127-354 overall | 28.35 mm | 73.35 deg | 25.09 mm | 33.32 deg |

Interpretation:

```text
The fixed local correction is real and useful, but it is not valid for the entire long run because the markerless tracker loses or changes state later in the video.
```

Specifically:

- Frames 127-170 confirm the original result very strongly.
- Frames 171-197 are partially improved, but rotation error grows.
- Frames 198-221 keep improved rotation but have poor translation, indicating position drift rather than just frame-offset error.
- Frames 222-260 again improve well, suggesting tracking recovers or enters a similar local frame.
- Frames 261-354 are not explained by the same correction. The tracker is likely no longer following the same plug pose reliably.

An all-frame correction was also tested. It did not solve the problem:

```text
all-frame correction rotation euler xyz = [-69.840 deg, 0.983 deg, 1.766 deg]
overall corrected translation = 28.61 mm
overall corrected rotation = 37.46 deg
```

This is worse than the early-window correction where tracking is stable. Therefore, the full-video result should not be described as one global systematic offset. A better description is:

```text
V7 has a stable local-frame offset during good tracking segments.
When tracking drifts or locks onto a different surface/state, a single correction cannot fix the whole sequence.
```

Practical next step:

```text
Use ArUco validation to detect good tracking segments and either:
1. report segment-level performance, or
2. add reinitialization/keyframes so the tracker does not drift into a different local state.
```

## Current Best Baselines

There are now two useful "best" baselines, depending on the evaluation target.

If the target is visual bbox placement and surface alignment, the best practical baseline is:

```text
V7: pose-gated RGB-D clustering + sampled STL + unconstrained point-to-point ICP
```

If the target is strict 6-DoF agreement with the ArUco-derived plug pose, the better baseline is:

```text
V5: pose-gated RGB-D clustering + locked rotation ICP
```

V5 preserves an orientation closer to the ArUco plug frame, while V7 gives better surface fit and slightly better plug-center translation.

Current output:

```text
outputs/phase3/cluster_assisted_track_127_170_v7_full_rotation/poses_cluster_assisted.csv
outputs/phase3/cluster_assisted_track_127_170_v7_full_rotation/cluster_assisted_overlay.mp4
outputs/phase3/cluster_assisted_track_127_170_v7_full_rotation/frames/
outputs/phase3/cluster_assisted_track_127_170_v5_pose_gate/poses_cluster_assisted.csv
outputs/phase3/cluster_assisted_track_127_170_v5_pose_gate/cluster_assisted_overlay.mp4
```

Open the video:

```powershell
start outputs\phase3\cluster_assisted_track_127_170_v7_full_rotation\cluster_assisted_overlay.mp4
```

Overlay color convention:

```text
blue: selected depth mask
orange: initial pose before ICP
green: accepted/final ICP pose
magenta: ArUco-derived plug reference, when available
yellow rectangle: broad ROI used before pose gating
```

Precise meaning of orange:

```text
Orange is always the pre-ICP pose for that frame.

However, after frame 127, orange is usually not a completely fresh
depth/PCA pose. It uses the current depth cluster for translation, but
inherits orientation from the previous accepted pose.

In V5/V6, the previous orientation came from locked-rotation tracking.
In V7, the previous orientation came from full unconstrained ICP, so it
could follow the plug rotation better.
```

## How To Reproduce Key Experiments

Inspect data:

```powershell
python scripts/inspect_bag.py data/raw/plug_with_tag_1280.bag
python scripts/inspect_bag.py data/raw/plug_without_tag_1280.bag
python scripts/view_stl.py data/raw/plug.STL
```

Run ArUco marker pose:

```powershell
python scripts/run_aruco_pose.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --marker_length 0.011 ^
  --marker_id 32 ^
  --out_csv outputs/phase2/poses_with_tag_marker.csv ^
  --out_video outputs/phase2/with_tag_aruco_overlay.mp4
```

Run ArUco-derived plug pose:

```powershell
python scripts/run_tagged_plug_pose.py ^
  --config config.yaml ^
  --marker_csv outputs/phase2/poses_with_tag_marker.csv ^
  --out_csv outputs/phase2/poses_with_tag_plug.csv
```

Generate frame review sheets:

```powershell
python scripts/export_frame_review.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --out outputs/phase3/frame_review_with_tag
```

Debug depth clusters for frame 127:

```powershell
python scripts/debug_depth_clusters.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --config config.yaml ^
  --roi_key cluster_roi_tagged ^
  --frame_id 127 ^
  --out_dir outputs/phase3/depth_clusters_frame127_select6 ^
  --select_labels 6
```

Run current V7 tagged validation:

```powershell
python scripts/run_cluster_assisted_tracker.py ^
  --bag data/raw/plug_with_tag_1280.bag ^
  --stl data/raw/plug.STL ^
  --config config.yaml ^
  --roi_key cluster_roi_tagged ^
  --reference_csv outputs/phase2/poses_with_tag_plug.csv ^
  --out_dir outputs/phase3/cluster_assisted_track_127_170_v7_full_rotation ^
  --start_frame 127 ^
  --max_frames 44 ^
  --disable_icp_acceptance
```

The same tracker can be run on the untagged bag by changing:

```text
--bag data/raw/plug_without_tag_1280.bag
--roi_key cluster_roi_without_tag
```

and omitting `--reference_csv`.

## Main Issues Found

### Depth Is Incomplete

The observed point cloud is only the visible surface. The back side and much of the volume are missing. This makes full 6-DoF pose underconstrained.

### Rectangular Shape Is Ambiguous

The plug is close to a long rectangular box. A partial surface can fit well under multiple rotations. This is why full ICP can produce high fitness while disagreeing with ArUco orientation.

### Masks Can Be Good But Pose Can Still Be Wrong

V5 showed that even a good mask does not guarantee good pose. The pose prior, orientation update, and ICP acceptance matter separately.

### Previous Masks Can Become Contaminated

Previous-mask tracking worked until it began including hand/wire/connector pixels. After contamination, it reinforced the wrong object. STL/bbox gating is more reliable than raw previous masks.

### ArUco Reference Is Useful But Approximate

The ArUco marker gives a strong reference for tagged frames, but final plug pose depends on the manually tuned marker-to-plug transform. Therefore, ArUco comparison should be treated as a reference check, not perfect ground truth.

## Suggested Next Steps

1. Run V7 on the untagged sequence and save the same diagnostics.

2. Add a bounded-rotation variant:

```text
Allow ICP rotation, but reject sudden 80-90 degree flips unless there is overwhelming evidence.
```

3. Add keyframe reinitialization:

```text
Every N frames, recompute rough orientation from cluster/PCA or from a manually selected good frame.
```

4. Try point-to-plane ICP only after better normals are available:

```text
Point-to-plane may help surface alignment, but sparse depth holes may make normals unreliable.
```

5. Consider edge/silhouette scoring:

```text
Use RGB image edges and projected STL bbox/silhouette to choose between rotation hypotheses.
```

6. Report V7 honestly:

```text
The method gives useful visual localization and plug-center tracking,
but full coordinate-frame orientation remains ambiguous under partial RGB-D observations.
```

## Current Takeaway

The project has reached a meaningful classical baseline. It can:

- read RealSense RGB-D data,
- estimate ArUco marker and approximate plug reference pose,
- segment a usable visible plug surface from depth/color,
- initialize pose without ArUco on a good frame,
- track through a short tagged sequence,
- align the STL with ICP,
- save per-frame pose CSVs and diagnostic overlays,
- expose clear failure modes for future improvement.

The strongest current result is not a perfect 6-DoF plug tracker. It is a well-documented classical pipeline that gets the plug bbox into the right image region and shows exactly why true 6-DoF markerless tracking is hard with the provided depth data.
