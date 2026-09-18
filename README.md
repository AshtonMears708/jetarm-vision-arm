# JetArm Vision Arm — Show It an Object, It Fetches the Match

A vision-guided robotic arm that takes its instruction from what you hold up to the camera.

Show the arm a **fork**, and it finds and picks up the **red** block. Show it a **knife**, it
picks the **green** one. A **spoon** gets the **blue** one. The object you present is the command.

Built on a HiWonder JetArm (Jetson Orin Nano 8GB) under ROS 2.

---

## How it works

The system runs two distinct perception stages against the same camera.

**Stage 1 — read the instruction.**
A YOLO classifier watches for a utensil held up to the camera. A detection only counts once the
same class has appeared in **20 consecutive frames above 0.65 confidence**, which suppresses the
single-frame false positives that would otherwise send the arm after the wrong block. The
resulting class maps to a target color.

**Stage 2 — find and grasp the target.**
The arm returns to its table-facing home pose and segments the scene for the target color:

1. `get_top_surface()` isolates block faces. L2-gradient Canny edges are dilated and then
   inverted, and that mask is AND-ed with an adaptive Gaussian threshold — so any pixel near an
   edge is excluded and only flat interior surfaces survive.
2. The masked frame converts to **LAB color space**, which separates chroma from luminance and
   holds up far better than RGB under changing room light.
3. `decide_pick_order()` selects the largest matching detection by bounding-box area, treating
   size as a proxy for confidence.
4. `pixel_to_world()` converts image coordinates to arm coordinates.
5. The world position goes to the arm's inverse-kinematics service
   (`kinematics/set_pose_target`) for approach, descent, grasp, lift, and place.

---

## Engineering notes

Things that were harder than they look, and how they were solved.

**The depth camera failed, so depth was reconstructed from geometry.**
The original design used an Orbbec Gemini Plus for 3D block positions. When the depth stream
became unreliable, the fix was not to abandon 3D but to recover it a different way: blocks sit on
a known table plane, so a single RGB pixel plus a calibrated plane is enough to solve for world
position. `pixel_to_world()` applies an extrinsic plane shift, inverse lens-distortion mapping for
USB cameras, a projection matrix, and finally YAML-driven scale and offset correction.

**IK pitch tolerance is held constant across every motion phase.**
Letting the solver pick freely at each phase meant it could return different arm configurations
for approach versus descent, producing sudden elbow and wrist flips mid-motion. Pinning
`IK_PITCH_TOL` to the same range everywhere forces a consistent solution family.

**The wrist-roll servo is deliberately omitted from the transit pose.**
Commanding all five servos while carrying a block would reset the gripper roll and drop or
reorient whatever was held. The transit pose commands servos 1–4 only.

**Threading.** Arm motion blocks. Camera intake cannot. The node runs on a ROS 2
`MultiThreadedExecutor` with reentrant callback groups and a frame queue.

---

## Results, honestly

The pick-and-place loop works end to end: utensil recognized, correct color selected, block
located, grasped, and placed at the mapped drop position.

**The custom-trained classifier lost to the stock one.** A YOLO classification model was trained
on a hand-collected 927-image utensil dataset, all photographed for this project. Benchmarked
against stock YOLOv11 weights in live testing, the custom model performed worse, so the stock
weights ship in `src/cv_control.py`. The training notebook is kept in `ml/` because the negative
result is part of the work.

**Auditing the dataset explained the negative result.** A later audit of the 927 images turned up
two problems that together account for most of the gap:

1. *A misfiled capture session.* All 81 images in `IMG_5697`–`IMG_5777` were forks sitting in the
   spoon class — 8.7% of the dataset, and 22% of everything labeled "spoon." Corrected class
   counts are **393 forks, 245 knives, 289 spoons** (the original figures of 312/245/370 were
   wrong). Found by cross-validating a silhouette-shape classifier over the images and then
   reviewing the disagreements and a random sample by hand; the filenames group into eight
   contiguous capture sessions, and one had been dropped into the wrong folder.
2. *Background–class leakage.* Each class was shot in its own setting, so the background alone
   nearly gives away the label. A classifier trained on border pixels only — every utensil pixel
   excluded — reaches **79% accuracy against a 40% baseline**. A model trained on this data can
   score well by learning the table rather than the utensil, then degrade in live testing where
   the background is the robot's actual workspace. That is exactly the failure that was observed.

The corrected labels ship with the dataset. Any retrain should split by capture session rather
than at random, so train and validation folds do not share a background.

A separate fruit classifier was trained by transfer learning on the public Fruits-360 dataset
(3 classes, 224×224, torchvision backbone, rotation/flip/color-jitter augmentation) and deployed
to the Jetson as TorchScript. See `ml/fruit-classifier-transfer-learning.ipynb`.

---

## Layout

```
src/          Final system. cv_control.py is the main ROS 2 node.
              arm_api.py, jetarm_control.py, cv_pipeline.py are supporting modules.
experiments/  Earlier iterations, kept for history. Color-only sorting, the fruit
              variant, and the standalone pick-and-place test.
ml/           Model training notebooks (Kaggle, GPU).
docs/         HiWonder's ROS 2 launch-command reference for this arm (vendor documentation,
              not authored here).
```

## Running it

Requires a HiWonder JetArm with the vendor ROS 2 stack, plus `ultralytics`, `opencv-python`,
`numpy`, and `pyyaml`. Calibration files (`transform.yaml`, `calibration.yaml`) are hardware
specific and live on the arm at `/home/ubuntu/ros2_ws/src/app/config/`.

```bash
ros2 launch sdk jetarm_sdk.launch.py   # vendor SDK first
python3 src/cv_control.py
```

Model weights are not committed — they are large binaries and hardware specific.

---

## Credits

Team project at Western Carolina University, spring 2026.
Built with **Wyatt** and **Kai**.
