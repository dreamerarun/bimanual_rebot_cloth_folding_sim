# Reproducibility

This checklist lets you verify the project from a clean clone: the demo needs
**no checkpoint and no dataset**, and every number below was measured on the
reference machine.

## 0. Reference environment

| Item | Value |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 4050 Laptop, 6 GB VRAM, driver 580.x |
| Python | 3.11.15 (conda env `leisaac`) |
| Isaac Sim | 5.1.0.0 (`pip install "isaacsim[all]==5.1.0.0"`) |
| torch | 2.7.0+cu128 (bundled with Isaac Sim) |
| lerobot | 0.4.2 |
| numpy / Pillow | 1.26.0 / 11.3.0 |
| Display | X11 `:1`, 2560×1600 (GUI runs use a 960×540 window) |

Install:

```bash
conda create -n leisaac python=3.11 -y && conda activate leisaac
pip install "isaacsim[all]==5.1.0.0"
pip install -r requirements.txt
```

Every Isaac Sim script must be launched with

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH ~/miniconda3/envs/leisaac/bin/python -u <script>
```

to avoid picking up the host Python's incompatible libraries.

## 1. Assets

```bash
python scripts/fetch_assets.py
```

* Clones `https://github.com/Seeed-Projects/reBot-Isaacsim.git` into
  `assets/rebot_t701/` (~436 MB).
* Expected check: `assets/rebot_t701/urdf/RS/urdf/ReBot_Arm_RS.urdf` exists.
* `assets/` is gitignored; if the URDF is missing, importing
  `wam_fold.rebot_kinematics` fails fast with an explanatory error.

Build the USD (one-time, needs Isaac Sim):

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  ~/miniconda3/envs/leisaac/bin/python scripts/02_build_rebot_usd.py
```

* Expected check: `assets/rebot_t701/usd/rebot_b601_flat.usd` exists and reports
  **6 revolute + 2 prismatic** joints, with a valid `/ReBot_Arm_RS` prim. The
  importer writes the geometry payload next to it as
  `usd/configuration/rebot_b601_flat_base.usd` (~95 MB); the env references
  `/ReBot_Arm_RS` explicitly, so the wrapper needs no default prim.

## 2. Quick check — static (no GPU)

```bash
python -m compileall -q wam_fold scripts scene && echo COMPILE_OK
python - <<'PY'
import wam_fold.config as c
print("cloth resolution", c.ClothCfg().resolution)
print("cloth size", c.ClothCfg().size)
PY
```

Both must succeed after step 1 (the kinematics module imports the URDF).

## 3. Quick check — scripted expert (single episode)

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  ~/miniconda3/envs/leisaac/bin/python -m wam_fold.run_episodes \
  --episodes 1 --out /tmp/repro_raw --max-steps 340 \
  --cloth-jitter 0.02 --seed 0
```

Expected: `success=True`, `fold_error=0.00 cm` (the fold error is deterministic;
the edge gap varies by a few cm run to run). Exit code 0, one pickle written to
`/tmp/repro_raw/`.

## 4. Quick check — GUI replay of the recorded success

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  ~/miniconda3/envs/leisaac/bin/python -u scripts/09_replay_gui.py \
  outputs/actions_ep2.npz --no-obs-cameras \
  --frames-dir /tmp/fold_frames --video /tmp/fold_ep2.mp4 \
  --hold-seconds 60
```

Expected console lines:

```
[replay] 340 actions ...
[replay] recorded fold_error=0.00 cm → replay delta=0.00 cm
[replay] DONE success=True fold_error=0.00 cm
```

Expected artifacts: 340 JPEGs in `/tmp/fold_frames`, `/tmp/fold_ep2.mp4`
(960×540, 20 fps, ~17 s). Compare against `outputs/fold_ep2_gui.mp4`.

## 5. Released checkpoint — closed-loop policy (no dataset needed)

```bash
curl -L -O https://github.com/dreamerarun/rebot-cloth-fold/releases/download/v1.0-smolvla-run3/smolvla_run3_pretrained_model.tar.gz
tar -xzf smolvla_run3_pretrained_model.tar.gz -C /tmp
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  ~/miniconda3/envs/leisaac/bin/python -u scripts/08_eval_smolvla.py \
  --ckpt /tmp/smolvla_run3 --stats outputs/smolvla_run3_stats.json \
  --episode-index 1 --max-steps 340 --save-actions /tmp/actions_ep1.npz
```

Expected: `success=True`, `fold_error=0.00 cm` in most episodes; edge gap
0.7–5.4 cm depending on the initial cloth pose. Policy inference runs on CPU
(~1.6 s per 25-step chunk); add `--gui --clean-ui` to watch.

## 6. Full pipeline (optional, hours)

1. **Collect 60+ jittered episodes** — one fresh process per episode:
   `bash scripts/collect_batch.sh 0 60 0.02`
   Expected: all `success=True`; failed episodes are discarded by default.
2. **Convert**: `scripts/07_convert_official_lerobot.py --raw data/raw --out data/lerobot_official`.
   Expected: 62 episodes / 21,080 frames / 20 fps / 2 cameras (top, front) /
   14-D state and action.
3. **Train** with the command in README step 3.
   Expected: ~0.09 s/step, loss ≈ 0.011 after 24k steps, 100 M trainable of
   450 M params, peak VRAM under 6 GB.
4. **Evaluate** with `scripts/08_eval_smolvla.py` (README step 4).
   Expected on the reference checkpoint: 6/6 successes, fold error 0.00 cm in
   most episodes, edge gaps 0.7–5.4 cm.

## Determinism notes

* PhysX deformable cloth is **not** restorable, so batch collection and
  evaluation use one episode per process (`wam_fold/run_episodes.py` and
  `08_eval_smolvla.py` spawn / expect a single episode).
* The fold *error* is stable at 0.00 cm; the *edge gap* is contact-sensitive and
  varies by 1–3 cm between runs and between record and replay.
* The replay in step 4 is open-loop: no model runs in the loop, so it is exactly
  as smooth as the saved trajectory and reproduces the recorded fold.
* Frames are not bit-identical across GPUs/driver versions (rasterisation and
  contact solvers differ); success metrics are the reproducible contract.

## Verification log (2026-11-13)

The checklist above was executed from a clean clone of
`https://github.com/dreamerarun/rebot-cloth-fold` (commit `b481b60`):

| Check | Result |
|---|---|
| `python -m compileall -q wam_fold scripts scene` | `COMPILE_OK` |
| `import wam_fold.config` | `PROJECT_ROOT` resolves to the clone, cloth 0.52×0.52 m, resolution 20 |
| `import wam_fold.rebot_kinematics` **without** assets | clear `FileNotFoundError` pointing at `scripts/fetch_assets.py` |
| same import **with** assets (symlinked to the fetched tree) | `dof 6`, TCP `gripper_end` |
| `02_build_rebot_usd.py` from a copy of the upstream `urdf/` tree | 6 revolute + 2 prismatic, `/ReBot_Arm_RS` valid, 95,480,663-byte payload (identical to the shipped build) |
| GUI replay of `outputs/actions_ep2.npz` | `success=True`, `fold_error=0.00 cm`, `delta=0.00 cm`, edge gap 4.6 ± 1.7 cm |
| replay artifacts | 340 JPEGs, MP4 960×540, 20 fps, 2.1 MB |

The 436 MB asset tree was symlinked rather than re-cloned during the check;
`scripts/fetch_assets.py` performs the real clone and produces the same paths.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CUDA out of memory` at startup | another Isaac Sim / eval process is alive; wait for `nvidia-smi --query-compute-apps` to be empty |
| `scene state is corrupted` | two GUI processes shared the GPU; kill all and relaunch |
| Eval hangs after loading weights | `HF_HUB_OFFLINE=1` missing (already set inside `08_eval_smolvla.py`) |
| Cloth metric reads stale points | fabric enabled; re-apply `/physics/fabricEnabled=False` + `SimulationManager.enable_fabric(False)` |
| URDF import error | assets not fetched — run `scripts/fetch_assets.py` |
