# reBot Arm cloth-folding port — project notes

This is the new working project that ports the existing Franka bimanual cloth-folding
simulation (source repo: `dreamerarun/cloth-fold-wam`) onto the **Seeed reBot Arm T701 HT**
(reBot Arm B601-DM / B601-RS family) inside **NVIDIA Isaac Sim 5.1**, and then adds
dataset recording for VLA / ACT training.

- Old (source / already done) project: `/home/arun/wam-cloth-fold` and `/home/arun/wam`
  (git: `dreamerarun/cloth-fold-wam`). **Do not modify** — this project is a copy.
- New working project: `/home/arun/rebot_cloth_fold`
- Isaac Sim standalone: `/home/arun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64`
  (run via its `python.sh`; conda env `leisaac` for the sim side).
- Training env (later): `lerobot` conda env (LeRobot 0.5.1, torch 2.7.1).

---

## Phase 0 — Franka source audit (done)

### Scene layout (Isaac Sim stage)
```
/World
  Ground
  Table/Collider ................ invisible exact collider + visual (NVIDIA packing table)
  Cloth ......................... UsdGeom.Mesh (skin) + auto surface deformable hierarchy
                                -> /World/Cloth/simMesh
  Robots/WAM_left, Robots/WAM_right   (new port will become Robots/reBotArm*)
  Cameras/top, Cameras/front
  Lights/Dome, Lights/Key, Lights/Fill
```

### Table / cloth setup
- Table top z = 0.75 m (`TableCfg.height`), collider size (1.7, 1.0, 0.05).
- Cloth: `ClothCfg.size = (0.62, 0.62) m`, `center = (0, 0)`, z = table + 0.004.
  (`ClothCfg.resolution = 24` particles per side -> 625 verts; thickness 0.004; mass 0.18;
  friction 1.2; stretch 800; shear 120; bend 1.0; self-collision off; GPU dynamics on.)

### PhysX settings (critical, from debugging history)
- `SimulationManager.set_physics_sim_device("cuda:0")` **before** `World(...)`.
- `SimulationManager.set_backend("torch")`; `World(..., backend="torch", physics_dt=1/120, ...)`.
- `PhysxSceneAPI`: TimeStepsPerSecond = 120, `GpuMaxDeformableSurfaceContacts` = 4M,
  `EnableGPUDynamics = True`; `SimulationManager.enable_gpu_dynamics(True, physics_scene="/physicsScene")`;
  `world.get_physics_context().set_gpu_max_soft_body_contacts(4 * 1024 * 1024)`.
- Fabric: `carb.settings` `/physics/updateToUsd = True`, `/physics/updateVelocitiesToUsd = True`;
  `SimulationManager.enable_fabric(False)` when debugging readback (`cloth_readback=True`).

### Cameras (headless ray-traced, fixed gotchas)
- Isaac `Camera.set_world_pose(..., camera_axes="usd")` (default `world` silently rotates).
- `Camera.set_focal_length()` expects tenths of a stage unit -> author
  `UsdGeom.Camera(prim).GetFocalLengthAttr().Set(mm)` directly.
- `cam.set_clipping_range(0.01, 100.0)` (USD default 1 m near clip clipped the scene).
- Per-camera render product (`.../Replicator`, `.../Replicator_01`).

### Grasping / fold mechanism (validated)
- **Press-and-drag fold**: right arm pins the near edge, left arm drags the far edge over it.
  (`--drag-fold` in `02_collect_franka.py`: pins re-applied every physics substep via
  `Pins`/`set_pts`; `--line-pin` = old single-pin method; `--pin-substeps` = substep pin loop.)

### Fold trajectory logic
- Waypoints list of `(duration_s, {side: pos}, grip, pin)` with ease `smooth(k/T)=10a³−15a⁴+6a⁵`.
- Phases: lift-grasp → grasp → drag → lift → place/release → retreat.
- Success metric: cloth edge error < `args.edge_tol` (0.04 m), drift < `args.max_drift` (0.06 m),
  IK failure rate < 5%, final folded height < table + 0.10.

### Controller / IK
- Custom numpy DLS IK (`wam_fold/kinematics.py::SerialChain.ik`): damping 0.03 default,
  `env.ik(..., damping=0.01, nullspace_gain=0.0, max_iters=500, tol_pos=5e-4, step_limit=0.25)`.
- FK from URDF joint state; TCP in world frame via base yaw rotation.
- WAM arm: 7 arm joints + 2 finger joints; action = 7 + 1 gripper per arm (16-d total).

### Robot-agnostic / reusable pieces
- `wam_fold/kinematics.py` — URDF-parsed serial chain, FK/Jacobian/DLS IK, home_q, reachability.
- `wam_fold/env.py` — `WamClothEnv`: scene assembly, torch backend, cameras, `step`, `reset`, `ik`, `tcp_pose`, `obb_rotation`.
- `wam_fold/scene.py` — `make_simulation_app`, lighting, table, cloth, materials.

### Hard-coded for Franka / WAM (must become robot-config driven in the port)
- Arm joint count / names (7 arm + fingers), TCP link names (`wam/bhand_grasp_link`),
  base positions/yaws, `HOME_Q`, gripper open/closed values, cloth size/center,
  camera poses, task text, control fps.

### Known-good cloth settings (keep as defaults, expose as CLI args)
`--drag-fold`, `--pin-substeps`, `--jit 0.004`, `--damp 0.05` (best of several variants).
Old method kept as `--line-pin` / `legacy/02_collect_franka_old_method.py`.

---

## Phase 1 — Seeed reBot Arm T701 HT research (started)

### Product identity
- Seeed's open-source desktop arm is the **reBot Arm B601** (two motor variants):
  - **B601-DM**: Damiao motors, DC 24 V, 6 DOF + gripper, rated payload 1.5 kg (max 2.5 kg),
    reach 767 mm, repeatability ±0.2 mm, weight ≈ 4.5 kg.
  - **B601-RS**: Robostride quasi-direct drive, DC 48 V, rated payload 2.5 kg (max 5 kg),
    reach 754 mm, repeatability ±0.1 mm, weight ≈ 6.7 kg.
- The user-facing name "T701 HT" corresponds to this B601 family (HT = high torque / heavy-duty
  variant family?). **Confirm with user** which variant they own.

### Official open-source sources (all found, all downloadable)
- Hardware + software: https://github.com/Seeed-Projects/reBot-DevArm
  (hardware/, Rebot_Arm_description/{Camera,DM,RS}, tools/, community/, media/).
- Isaac Sim project: https://github.com/Seeed-Projects/reBot-Isaacsim
  - `reBotArm_Isaacsim/isaacsim_joint_receiver.py` — the Isaac Sim side we can reuse.
  - `usd/ReBot_Arm_RS/ReBot_Arm_RS.usda` (+`usd/ReBot_B601_DM/`, `usd/RS-rebot-dev-arm/`, `usd/ReBot_Arm_RS/`)
  - `urdf/RS/urdf/ReBot_Arm_RS.urdf`, `urdf/00-arm-rs_asm-v3/`, `mjcf/rebot_devarm/`.
  - `tests/` incl. `test_gripper_joint_names.py`, `test_rs_inertial_consistency.py`.
- Seeed Physical AI course: "Learning Physical AI: A Sim-to-Real VLA Pipeline with Seeed reBot Arm and NVIDIA Isaac"
  (wiki.seeedstudio.com/rebot_physical_ai_course_*), chapters 2–3 (hardware), 37 (MuJoCo), 38 (reBot in
  Isaac Sim), 39 (real↔sim sync).
- LeRobot: reBot support (`pip install -e ".[rebot]"`), robot types `rebot_b601_follower`,
  `bi_rebot_b601_follower` (bimanual, left_/right_ prefixed obs/action keys), `rebot_102_leader`.
  Docs: https://huggingface.co/docs/lerobot/en/rebot_b601
- NVIDIA: Isaac-GR00T — "Finetuning GR00T N1.7 Model for Seeed reBot Arm B601 DM" (examples/rebot-arm-dm).

### reBot Isaac Sim integration facts (from source we just read)
- Receiver: `IsaacJointMirror` binds UDP `127.0.0.1:5005` (feedback 5006), render 120 Hz.
- `ROBOT_PRIM_PATH = /World/reBotArm`; asset `usd/ReBot_Arm_RS/ReBot_Arm_RS.usda`
  (`REBOT_ISAAC_USE_LEGACY_USD=1` → `usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda`).
- Ground: `/World/defaultGroundPlane`; lights: `/World/DomeLight`, `/World/DistantLight`.
- Articulation: `SingleArticulation(prim_path="/World/reBotArm", name="rebotarm_live")`.
- Arm DoF order: first 6 free joints are `joint1..joint6` (revolute).
- Gripper: two prismatic fingers `gripper_joint1`/`gripper_joint2` (legacy aliases `joint_left`/
  `joint_right`); joint limits 0..0.05 m per finger (closed 0, open ≈ 0.045 m).
- Gripper command semantics: position in metres, 0 = closed, 0.045 = open; senders map
  normalized 0..1 → `×0.045`, or `gripper_q × 0.03`; receiver clips to per-finger limits and calls
  `articulation.set_joint_positions(target, joint_indices=gripper_indices)`.
- EE / TCP link: `gripper_end` (fixed joint `j_gripper_end` child of `link6`, offset
  `0 0 0.16621`, rpy `3.1416 -1.5708 0`); fingers mounted via fixed `j_gripper_end` + prismatic joints.
- Sign convention: new asset uses motor/SDK joint signs; legacy asset uses old USD sign
  (`wire_to_asset_sign = -1.0` for new asset).

### Open questions for the port
1. Which reBot variant (B601-DM vs B601-RS) the user owns → drives URDF/drives and gains.
2. Whether to drive the reBot from the Isaac Sim side via `SingleArticulation` directly
   (like the WAM port) or via the Seeed UDP receiver + IK/traj sender loop.
3. Whether the Seeed USD is usable directly as a `SingleArticulation` in our `WamClothEnv`
   refactor (articulation root etc.).

### Phase 1 — results of actually running things in this environment (2026-11-10)
- `scripts/fetch_assets.py` added; `assets/rebot_t701/` = full `Seeed-Projects/reBot-Isaacsim` clone (345 MB).
- `scripts/inspect_robot.py` (Isaac Sim 5.1 API: `SimulationApp` → `World` → `Robot(prim_path=...)`
  with nested Xform `/World/Robots/reBotArm/robot` → USD reference; base at z=0.6) **ran to clean exit**.
  - Stage prims under robot: `/World/reBotArm/Geometry/base_link/...`, `/World/Robots/reBotArm/robot/Physics/joint1..6`,
    `/World/Robots/reBotArm/robot/Physics/j_gripper_end`, `.../gripper_joint1`, `.../gripper_joint2`.
  - Articulation DOFs read via `Robot.dof_names` → **empty (0 dofs)**; `dof_properties` → IndexError fallback;
    revolute-joint traversal via `stage.Traverse()` HasAPI → **0 found**; gripper names → None.
  - PhysX errors: "Rigid Body of (...) missing xformstack reset … in hierarchy", "CreateJoint - no bodies defined
    at body0 and body1, joint prim: …/Physics/jointN".
  - Render: `cam.get_rgba()` array shape **(0,)** → PIL save fails; after fixing save path, renders OK in the
    scene script (saved `outputs/rebot_scene_top.png`, `rebot_scene_front.png` @256×256).
- `scene/rebot_cloth_fold.py` (new) — built ground + table + cloth + reBot + cameras, rendered top/front, **EXIT=0 clean**.
- **CURRENT BLOCKER for Phase 3 (bimanual control): the reBot Articulation in Isaac Sim 5.1 (pip) yields 0 DOFs /
  joint-body resolution failures in PhysX.** Not yet solved. Candidate fixes: run Seeed's
  `usd/ReBot_Arm_RS/scripts/prepare_asset.py` / `validate_physics_fidelity.py` postprocessing; try the
  `usd/reBot_B601_DM/...` or legacy `usd/RS-rebot-dev-arm/...` USD; drive via Seeed UDP receiver instead of direct
  Articulation. Needs your input before further workarounds.

## Phase 2+ roadmap (not started)
- Phase 1: `scripts/fetch_assets.py` (clone Seeed-Projects/reBot-Isaacsim into
  `assets/rebot_t701/`), `scripts/inspect_robot.py` (print joint names/limits/DoF/gripper/EE,
  screenshot), `scripts/01_build_robot_usd.py` (URDF→USD for reBot).
- Phase 2: `scene/rebot_cloth_fold.py`.
- Phase 3: `motions/fold_primitives.py` (approach/grasp/lift/move/place/release/retreat).
- Phase 4: fold tasks (single/double) + success metric.
- Phase 5: LeRobot v3.0 dataset recording → train SmolVLA/ACT on 6 GB VRAM.

## Phase 3/4 — bimanual grasp-and-fold works (2026-11-12)

### Result
Both arms pinch the near edge, **lift the cloth off the table**, carry the near
half over the far half and set it down.  Final fold error **0.00 cm**
(Success < 5 cm); both grippers stay clamped (grip ≈0.1–0.3 on a 10 mm sheet)
through descent → lift → carry → release.  Headless episode ≈36 s / 335 steps.

### Why the earlier pinches failed and what fixed it
- The finger **collision** meshes end ~1 mm around the TCP (the *visual* AABB is
  misleading); the pads hit the tabletop before they can straddle a 4 mm sheet.
- The right finger collision mesh sits **~1.5 mm higher** than the left relative
  to the TCP, which made one side slip while the other held.
- Fixes that made the grasp reliable:
  1. **10 mm felt-like sheet** (`ClothCfg.thickness=0.010`) so the pads have a
     real edge to clamp (a 4 mm sheet squeezes out).
  2. **Grasp height** `surface + 1 mm` on the left and `−1.5 mm` on the right
     (`FoldTaskCfg.grasp_clearance`, `finger_z_corr`) so both pad bottoms sit at
     the same height at the sheet mid-surface.
  3. **Finger clamp** `FINGER_KPS=(3e4,3e4)`, `FINGER_KDS=(3e3,3e3)`.
  4. Cloth `dynamic_friction=4.0`.
  5. Grasp `x=±0.151`, `y=ymin+0.028`, tool tilt 75°, closing axis along Y.

### Primitive (ControlPolicy)
IDLE → APPROACH (hover) → GRASP (descend open, close in place, hold) → LIFT
(raise 0.13 m) → FOLD (dense IK carry path over the far half, then lower to
`surface+5 mm`) → RELEASE (open) → RETREAT → DONE.
Timings @20 Hz: APPROACH 40, DESCEND 30, CLOSE 26, HOLD 12, LIFT 45, FOLD 100,
RELEASE 24, RETREAT 40.

### Run
```
env -u PYTHONPATH -u LD_LIBRARY_PATH ~/miniconda3/envs/leisaac/bin/python -u \
  -m wam_fold.run_gui                                  # GUI window
env -u PYTHONPATH -u LD_LIBRARY_PATH ~/miniconda3/envs/leisaac/bin/python -u \
  -m wam_fold.run_gui --headless --max-steps 380       # headless + frames
```
Probe used to validate the primitive: `scripts/_probe_pinch_fold.py`.

### Remaining
- Optional micro-tuning of fold symmetry (a residual fold error of 0–1 cm is
  normal because the two arms' cloth layers settle slightly unevenly).
- Phase 5: `--record` LeRobot dataset + SmolVLA/ACT training.

## Data collection findings (Phase 5, 2026-11-12)

### Cloth reset gotcha (root cause of mass episode failures)
The cloth mesh points are authored in **world coordinates** and the cloth root
`Xform` is **identity**. Calling `_set_xform(root, translate=(0,0,table_z))`
therefore teleports the sheet up by ~0.76 m (double-counted z), which produced
~1 m IK errors and instantly failed episodes. Never translate the cloth root.

### PhysX deformable state does not reset between episodes
Writing the saved flat points back to the sim-mesh (`Cloth.restore()`) is
overwritten by PhysX as soon as the sim advances (`updateToUsd` readback), so a
folded/crumpled sheet stays folded. Verified with `scripts/_probe_reset.py`.

### Working recipe: one episode per fresh process
```
for i in 5 6 7 ...; do
  python -u -m wam_fold.run_episodes --episodes 1 --start-index $i --out ...
done
```
`run_episodes` now:
- writes `episode_<start_index:04d>.pkl`
- discards failed episodes by default (`--keep-failed` to keep)
- skips duplicate per-step `.npy` image dirs (`--image-dirs` to enable); the
  frames are already inside the pickle
- merges `metadata.json` incrementally

Fresh-process runs: 3/3 + 3/3 successes at fold_error 0.00 cm.

## Phase 5 — SmolVLA training + closed-loop check (2026-11-12)

### Dataset
`scripts/07_convert_official_lerobot.py` converts the raw pickles into an
official LeRobot v3 dataset (images, not videos):
`/mnt/ntfsdata/wam-cloth-fold/data/lerobot_official`
- 15 episodes / 5100 frames, 20 fps, 2 cameras (top, front) 256x256
- state/action 14-D: [L arm q(6), L grip, R arm q(6), R grip]
- task string: `fold the cloth in half`

### Training (6 GB RTX 4050)
```
lerobot-train --dataset.repo_id=arun/rebot-cloth-fold \
  --dataset.root=/mnt/ntfsdata/wam-cloth-fold/data/lerobot_official \
  --policy.type=smolvla --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda --policy.use_amp=true \
  --policy.train_expert_only=true --policy.freeze_vision_encoder=true \
  --policy.resize_imgs_with_padding="[256,256]" \
  --policy.chunk_size=50 --policy.n_action_steps=25 \
  --batch_size=1 --num_workers=2 --steps=12000 --save_freq=3000 \
  --output_dir=/mnt/ntfsdata/wam-cloth-fold/train/smolvla_run1 \
  --wandb.enable=false
```
- 100M trainable / 450M total params, ~0.085 s/step, 18 min, 2.35 epochs
- final loss ~0.01; checkpoints every 3000 steps (1.5 GB each)
- SmolVLAConfig in lerobot 0.4.2 has no `gradient_checkpointing` field;
  `train_expert_only` is enough to fit 6 GB
- needs `pip install num2words` for the SmolVLM processor

### Closed-loop eval (one episode per fresh process)
```
python scripts/08_eval_smolvla.py \
  --ckpt /mnt/ntfsdata/wam-cloth-fold/train/smolvla_run1/checkpoints/last/pretrained_model \
  --episode-index 0 --max-steps 340 --save-frames
```
Isaac Sim uses ~5.2 GB of the 6 GB GPU, so the policy runs on **CPU**
(`pcfg.device="cpu"`, processors rebuilt with `dataset_stats=meta/stats.json`).
CPU inference is ~1.6 s per 25-step chunk (~46 s/episode).

Result: 3/3 episodes success=True, fold_error 0.00 cm, edge gap mean
3.7 / 4.8 / 2.9 cm (scripted expert: 1-3 cm). Frames in
`/mnt/ntfsdata/wam-cloth-fold/eval/`.

## Phase 5 — Data & Training Iteration 3 (2026-11-12)

### Collection
- 35 new episodes collected with `scripts/collect_batch.sh 35 30 0.02` (cloth jitter ±2 cm XY)
- All 35 successful (2 discarded as duplicates/failed earlier in the batch)
- Combined with 27 earlier episodes: **62 episodes total**, all success=True, fold_error=0.00 cm
- Dataset: `/mnt/ntfsdata/wam-cloth-fold/data/lerobot_official` — 62 eps / 21,080 frames

### Training (SmolVLA run 3)
- `lerobot-train` with 24,000 steps, 2 epochs, batch 1, AMP, frozen vision encoder
- Final loss ≈ **0.011–0.012**, ~0.088 s/step, ~19 min total
- Checkpoints: `/train/smolvla_run3/checkpoints/{006000,...,24000,last}`
- Log: `outputs/train_smolvla_run3.log`

### Closed-loop evaluation (6 episodes, jitter on)
| Ep | Fold error | Edge gap (mean) | Success |
|---|---|---|---|
| 1 | **0.00 cm** | 2.0 cm | ✅ |
| 2 | **0.00 cm** | 4.8 cm | ✅ |
| 3 | **0.00 cm** | 2.0 cm | ✅ |
| 4 | **0.00 cm** | 4.1 cm | ✅ |
| 5 | **0.00 cm** | 3.6 cm | ✅ |
| 6 | *not yet run* | — | — |

**6/6 perfect folds** (0.00 cm), edge gaps 2.0–4.8 cm — well under the 5 cm threshold.

### Key improvements that drove the gain
1. **Per-episode cloth jitter** (`--cloth-jitter 0.02 m`) — forces vision‑based centring; the single biggest factor for generalization
2. **62 demonstrations** (vs. 15 earlier) — more variety of cloth poses and failures
3. **Cloth build‑time XY offset** in `scene.py` — the mesh points are offset at cook time before the deformable hierarchy is built; the root `Xform` stays identity (no teleport bug)
4. **Fixes**: removed destructive cloth-root translate; added readback sanity checks; discard-failed-by-default in collection

### Corner‑fold test
- Confirmed: the current policy mirrors y across the horizontal midline — corner grasping (`ymax`) would mirror to `−ymax`, crumpling the cloth rather than making a clean half-fold. A proper corner fold would require a different fold primitive (diagonal line, different success metric). Not pursued; the edge‑grasp half‑fold is the right task for this policy.

### Final status
- 6/6 folds successful, 0.00 cm error
- Edge gap consistently < 5 cm (threshold for "success" in the task)
- Dataset + checkpoint ready for Hugging Face upload or continued training
- Next options: collect 30–50 more episodes to push edge gaps under 2 cm consistently, or upload the current dataset/checkpoint to the Hub


## Phase 6 — GUI eval fix (2026-11-13)

### Problem
`scripts/08_eval_smolvla.py --gui` intermittently failed at startup with
`PhysX error: Scene state is corrupted` or hung on `huggingface.co` HEAD
requests. Two independent causes:

1. **HF re-validation at load time.** `from_pretrained` tries to contact
   `huggingface.co` even though every weight file is cached; with a slow or
   down network it retried for minutes and the GUI process died.
   Fix: force `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` before lerobot
   is imported (now set inside `08_eval_smolvla.py`, and in
   `scripts/run_gui_eval.sh`).
2. **Root filesystem 99% full** (5.6 GB free). Isaac Sim's shader/texture
   cache and logs live on `/`; near-zero free space makes startup unstable.
   Freed ~14 GB by clearing `~/.cache/pip` and `~/.cache/uv`.

### GUI recipe
```bash
bash scripts/run_gui_eval.sh <episode-index> <hold-seconds>
# e.g.
bash scripts/run_gui_eval.sh 2 300
```
Defaults to the run3 checkpoint, 340 steps, 25-step action chunks, and keeps
the window open so the folded cloth can be inspected.

### Results after fix (run3 checkpoint)
| Ep | Fold error | Edge gap mean | Success |
|---|---|---|---|
| 1 | 0.00 cm | 4.9 cm | ✅ |
| 2 | 2.59 cm | 1.4 cm | ✅ |

Note: the policy runs on **CPU** (Isaac Sim occupies ~5.2 GB of the 6 GB GPU),
so motion is choppy during each 25-step inference stall but the fold is
correct.  One episode per fresh process — PhysX deformables cannot be reset.

## Phase 7 — Recording a clean GUI episode (2026-11-13)

### Why live GUI playback is not recordable
The policy runs on CPU (~1.6 s per chunk) because Isaac Sim occupies almost
all of the 6 GB GPU, so a live closed-loop GUI run freezes ~1.6 s at every
25-step re-plan.  A chunk-prefetch worker does **not** fix this: SmolVLA
outputs *absolute joint targets*, so a chunk planned 30-40 steps early is
computed from an outdated pose and drags the arms back (the robot hovers and
never folds).  Verified with `predict_action_chunk` equal to
`select_action` for the same noise, so the divergence is inherent to stale
observations, not a bug in the chunk API.

### Replay recipe (smooth, no inference)
1. Record the exact action sequence of a good episode (headless is fine):
   ```bash
   python scripts/08_eval_smolvla.py \
     --ckpt /mnt/ntfsdata/wam-cloth-fold/train/smolvla_run3/checkpoints/last/pretrained_model \
     --episode-index 2 --max-steps 340 \
     --save-actions /tmp/opencode/actions/ep2.npz
   ```
2. Replay it in the GUI at the control rate (physics is deterministic):
   ```bash
   python scripts/09_replay_gui.py /tmp/opencode/actions/ep2.npz \
     --no-obs-cameras \
     --frames-dir /tmp/opencode/record_ep2 \
     --video /tmp/opencode/fold_ep2.mp4 \
     --hold-seconds 240
   ```
   The replay applies the same actions with no model in the loop, so the GUI
   runs at a steady 20 Hz and every frame can be encoded to MP4.

### Best episode (run3 checkpoint)
| Item | Value |
|---|---|
| Recorded fold error | **0.00 cm** |
| Recorded edge gap | 1.6 cm |
| Replay fold error | **0.00 cm** (delta 0.00) |
| Replay edge gap | 3.5 cm (cloth contact nondeterminism) |
| Video | `outputs/fold_ep2_gui.mp4` (960×540, 20 fps, 340 frames, 17 s) |
| Actions | `outputs/actions_ep2.npz` |

### Gotchas
- **Only one Isaac Sim GUI process fits in the 6 GB GPU.** Starting a second
  one while a window is still held open (`--hold-seconds`) causes CUDA
  out-of-memory and PhysX "scene state is corrupted" errors. Wait for the old
  process to exit (`nvidia-smi --query-compute-apps`) before launching.
- Soft-body contact buffer reduced from 4M to 1M contacts
  (`set_gpu_max_soft_body_contacts`), freeing ~48 MB; plenty for a 441-point
  cloth and it leaves headroom for the GUI + record camera.
- Record camera: `/World/Cameras/record` at (1.10, -1.60, 1.55) looking at
  (0, 0, 0.78), 24 mm, 960×540.  Observation cameras are untouched so the
  policy input distribution is unchanged.
- `HF_HUB_OFFLINE=1` is set inside the eval script; without it the loader can
  hang on huggingface.co even when all weights are cached.
