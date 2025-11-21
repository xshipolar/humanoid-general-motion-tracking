<!-- Copilot / AI agent helper for contributors working in this repo -->
# Copilot instructions — humanoid-general-motion-tracking

This file contains concise, repo-specific guidance for AI coding agents (Copilot-style) to be productive in this repository.

1) Big picture
- **Purpose:** Lightweight MuJoCo-based toolkit for testing a pretrained whole-body motion-tracking policy on the Unitree G1 robot.
- **Primary scripts:** `sim2sim.py` (runs policy in MuJoCo, uses a JIT-ed torch policy), `view_motion.py` (kinematic-only visualizer).
- **Core data flow:** pickled motion files in `assets/motions/` -> `utils/motion_lib.MotionLib` constructs time-series tensors (root_pos/rot, dof_pos, velocities) -> `HumanoidEnv` queries `MotionLib.calc_motion_frame` to form mimic_obs -> policy (torch.jit) consumes concatenated obs -> outputs joint targets -> PD control computed in `sim2sim.py` -> MuJoCo simulation steps.

2) Key files & patterns to reference
- `sim2sim.py`: authoritative runtime. Pay attention to `HumanoidEnv.__init__` for robot-specific constants (stiffness, damping, `default_dof_pos`, `num_dofs`, control timestep `sim_dt`, `sim_decimation`). Observation assembly happens inside the loop (see `obs_prop`, `mimic_obs`, `obs_hist`). Note the policy is loaded via `torch.jit.load`.
- `view_motion.py`: shows how `MotionLib` is used to set `qpos` for purely kinematic visualization (good for debugging motions without policy).
- `utils/motion_lib.py`: implements motion loading & frame interpolation (`calc_motion_frame`). Motion files are pickles expected to contain keys `fps`, `root_pos`, `root_rot`, `dof_pos`.
- `utils/torch_utils.py`: quaternion helpers and `slerp` implemented with `torch.jit.script`. Preferred for performance; keep types compatible when calling these utilities.

3) Running & environment notes (extracted from README)
- Recommended env (example):
  - `conda create -n gmt python=3.8 && conda activate gmt`
  - `pip3 install torch torchvision torchaudio`
  - `pip install "numpy==1.23.0" pydelatin tqdm opencv-python ipdb imageio[ffmpeg] mujoco mujoco-python-viewer scipy matplotlib`
- Run examples:
  - Policy-in-the-loop simulation: `python sim2sim.py --robot g1 --motion_file walk_stand.pkl`
  - Kinematic viewer: `python view_motion.py --motion_file walk_stand.pkl`
- Note: README uses `--motion` in an example but the scripts accept `--motion_file`. Use `--motion_file` when running the code.

4) Project-specific conventions and gotchas
- **Motion file format:** Motion files are pickles with arrays `root_pos` (N x 3), `root_rot` (N x 4 quaternion), `dof_pos` (N x num_dof), and `fps`.
- **Smoothing:** `MotionLib` applies a box filter of width 19 frames to velocities — when altering velocity computations keep this smoothing in mind.
- **Time units & looping:** `MotionLib.calc_motion_frame` handles looping; time inputs are in seconds and `sim2sim` uses `control_dt = sim_dt * sim_decimation`.
- **Policy interface:** policy is a TorchScript module expecting a single `FloatTensor` of shape `(1, obs_dim)` and returns a flat action vector; code loads it with `torch.jit.load(policy_path, map_location=self.device)`.
- **Device handling:** scripts default `device = "cuda" if torch.cuda.is_available() else "cpu"`; when editing, keep `.to(device)` and `map_location` semantics.
- **Robot params are hard-coded in `sim2sim.py`** under `if robot_type == "g1"` — to add a new robot, add a new branch with consistent DOF ordering and update `assets/robots/` with the corresponding URDF/XML.

5) Examples of useful edit intents for agents (how to change things safely)
- Add a new motion: place a pickled motion file in `assets/motions/` (same keys), then run `python sim2sim.py --motion_file new_motion.pkl` to test.
- Change PD gains or limits: modify `stiffness`, `damping`, `torque_limits` arrays in `HumanoidEnv.__init__` and run short simulations to validate.
- Debug obs shaping: instrument `obs_buf` creation in `sim2sim.py` (print shapes and an example tensor) or use CPU device to iterate faster without GPU.
- Add unit tests: small tests can load a short motion via `MotionLib` and assert `calc_motion_frame` shapes/types; keep tests minimal and deterministic (use CPU device).

6) Integration points & external dependencies
- MuJoCo Python API + `mujoco_viewer` for rendering and stepping (`mujoco.MjModel`, `mujoco.MjData`, `mujoco.mj_step`, `mujoco.mj_forward`).
- Torch and TorchScript for policy and quaternion ops (`torch.jit.script` used extensively in `utils/torch_utils.py`).
- Motion files are repo-local pickles — there is no external dataset downloader in this repo.

7) Short examples to cite in PRs or edits
- When changing observation composition, reference the concat order in `sim2sim.py` around `obs_prop` and `obs_buf` (mimic_obs first, then proprio, then history).
- When adding a robot, mirror how `g1` sets `num_dofs`, `dof_names`, `default_dof_pos`, and camera distance for the viewer.

8) What not to change without verification
- Don't change the tensor/device contracts for `MotionLib.calc_motion_frame` (it returns torch tensors used by other code).
- Avoid removing `torch.jit.script` wrappers in `utils/torch_utils.py` without performance regression testing.

If anything here is unclear or you want more specific examples (small tests, extra run commands, or a suggested PR to add CI), tell me which area to expand and I will iterate.
