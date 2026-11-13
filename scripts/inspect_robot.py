#!/usr/bin/env python
"""Inspect the Seeed reBot Arm T701 HT (B601-DM / B601-RS) URDF via Isaac Sim.

Uses Isaac Sim 5.1 python API (World + Robot + Stage) and Isaac Sim's URDF
importer (`URDFParseAndImportFile`) -- the same route that produced a working
articulation for the Barrett WAM.

Prints:
  - imported prim path
  - articulation / prim info
  - free-DoF count + names (expect 6 arm joints + 2 gripper fingers = 8)
  - joint limits (if available)
  - gripper joint names / indices / limits
  - screenshot

Run:
    env -u PYTHONPATH -u LD_LIBRARY_PATH \
      ~/miniconda3/envs/leisaac/bin/python scripts/inspect_robot.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

PROJECT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

ARM_JOINT_COUNT = 6
GRIPPER_JOINT_NAMES = ("gripper_joint1", "gripper_joint2")
LEGACY_GRIPPER_JOINT_NAMES = ("joint_left", "joint_right")
DT = 1.0 / 120.0


def _joint_limits(prim):
    """Return (lower, upper) for a joint prim (USD >= 5.1 uses LimitAPI attrs)."""
    lo_a = prim.GetAttribute("physics:lowerLimit")
    hi_a = prim.GetAttribute("physics:upperLimit")
    lo = lo_a.Get() if lo_a and lo_a.IsValid() else None
    hi = hi_a.Get() if hi_a and hi_a.IsValid() else None
    if lo is None and hi is None:
        return None
    return (float(lo) if lo is not None else None, float(hi) if hi is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=str(PROJECT / "assets" / "rebot_t701" / "urdf" / "RS" / "urdf" / "ReBot_Arm_RS.urdf"))
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--out", default=str(PROJECT / "outputs" / "inspect_robot.png"))
    args = ap.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": not args.gui, "hide_ui": args.gui, "width": 640, "height": 480})

    from isaacsim.core.api import World
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.prims import is_prim_path_valid
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from wam_fold.scene import _set_xform
    from wam_fold.env import look_at_quat, _to_np

    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.set_physics_sim_device("cuda:0")
    SimulationManager.set_backend("torch")

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 120.0,
        rendering_dt=1.0 / 30.0,
        backend="torch",
    )
    stage = get_current_stage()

    # ---- reference the converted reBot USD (built by scripts/02_build_rebot_usd.py)
    robot_parent = UsdGeom.Xform.Define(stage, "/World/Robots/reBotArm")
    _set_xform(robot_parent.GetPrim(), translate=(0.0, 0.0, 0.6), rotate_z=0.0)
    robot_inner = UsdGeom.Xform.Define(stage, "/World/Robots/reBotArm/robot")
    robot_inner.GetPrim().GetReferences().AddReference(str(PROJECT / "assets" / "rebot_t701" / "usd" / "rebot_b601_flat.usd"), "/ReBot_Arm_RS")
    print(f"[inspect] robot prim children: {[str(c.GetPath()) for c in robot_inner.GetPrim().GetChildren()]}", flush=True)

    # ---- locate the articulation root (ArticulationRootAPI)
    root = None
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            root = str(p.GetPath())
            break
    print(f"[inspect] ARTICULATION ROOT = {root}", flush=True)

    robot = Robot(prim_path=root, name="rebotarm_inspect")
    world.scene.add(robot)
    world.initialize_physics()
    world.step()
    robot.initialize()
    world.step()

    dof_names = list(robot.dof_names)
    lower = np.zeros(max(len(dof_names), 1))
    upper = np.zeros(max(len(dof_names), 1))
    props = None
    # limits: read them from the USD joints (robot.dof_properties is torch-CUDA and
    # does an eager numpy conversion internally in Isaac Sim 5.1)
    usd_limits = {}
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint):
            lim = _joint_limits(p)
            if lim is not None:
                usd_limits[p.GetName()] = lim
    for i, name in enumerate(dof_names):
        if name in usd_limits:
            lower[i], upper[i] = usd_limits[name]
    try:
        props = robot.dof_properties
        lower = _to_np(props["lower"]).astype(np.float64)
        upper = _to_np(props["upper"]).astype(np.float64)
    except Exception as e:
        print(f"[inspect] robot.dof_properties unavailable ({e}); using USD joint limits", flush=True)

    print("=" * 72, sep="", flush=True)
    print("reBot Arm T701 HT (B601) - Isaac Sim inspection", flush=True)
    print("=" * 72, sep="", flush=True)
    print(f"[inspect] urdf asset : {args.urdf}", flush=True)
    print(f"[inspect] prim path   : {root}", flush=True)
    print(f"[inspect] num dofs     : {len(dof_names)}", flush=True)
    print(f"[inspect] dof names   : {dof_names}", flush=True)
    print(f"[inspect] arm dofs[0:{ARM_JOINT_COUNT}] : {dof_names[:ARM_JOINT_COUNT]}", flush=True)
    print(f"[inspect] dof limits (lower): {np.round(lower, 4).tolist()}", flush=True)
    print(f"[inspect] dof limits (upper): {np.round(upper, 4).tolist()}", flush=True)

    gripper_names = next(
        (gn for gn in (GRIPPER_JOINT_NAMES, LEGACY_GRIPPER_JOINT_NAMES) if all(x in dof_names for x in gn)),
        None,
    )
    gripper_idx = None if gripper_names is None else [dof_names.index(x) for x in gripper_names]
    print(f"[inspect] gripper names: {gripper_names} indices={gripper_idx}", flush=True)
    if gripper_names is not None and gripper_idx is not None:
        for name, i in zip(gripper_names, gripper_idx):
            lo = float(lower[i]) if len(lower) > int(i) else 0.0
            hi = float(upper[i]) if len(upper) > int(i) else 0.0
            print(f"[inspect]   {name}: lower={lo:+.5f} upper={hi:+.5f} (metres, closed=0, open=~0.045)", flush=True)

    rev = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    fixed = [p for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint)]
    prism = [p for p in stage.Traverse() if p.IsA(UsdPhysics.PrismaticJoint)]
    print(f"[inspect] revolute joints : {len(rev)}", flush=True)
    for p in rev:
        j = UsdPhysics.RevoluteJoint(p)
        ax = j.GetAxisAttr().Get() if j else None
        lim = _joint_limits(p)
        print(f"[inspect]    {p.GetPath()} axis={ax} limit={lim}", flush=True)
    print(f"[inspect] prismatic joints: {len(prism)}", flush=True)
    for p in prism:
        j = UsdPhysics.PrismaticJoint(p)
        ax = j.GetAxisAttr().Get() if j else None
        lim = _joint_limits(p)
        print(f"[inspect]    {p.GetPath()} axis={ax} limit={lim}", flush=True)
    print(f"[inspect] fixed joints    : {len(fixed)}", flush=True)
    for p in fixed:
        print(f"[inspect]    {p.GetPath()}", flush=True)

    from isaacsim.sensors.camera import Camera

    cam = Camera(prim_path="/World/inspect_cam", resolution=(640, 640), frequency=30)
    cam.initialize()
    cam.set_world_pose(
        position=np.array([0.6, 0.5, 0.7], dtype=np.float32),
        orientation=look_at_quat([0.6, 0.5, 0.7], [0.0, 0.0, 0.3]),
        camera_axes="usd",
    )
    try:
        from pxr import UsdGeom as _G

        _G.Camera(cam.prim).GetFocalLengthAttr().Set(18.0)
    except Exception as e:
        print("[inspect] focal fallback:", e, flush=True)
    cam.set_clipping_range(0.01, 100.0)
    for _ in range(6):
        world.step(render=True)
    rgba = cam.get_rgba()
    arr = np.asarray(rgba)[..., :3].astype(np.uint8)
    print(f"[inspect] render array shape: {arr.shape} dtype={arr.dtype}", flush=True)
    if arr.size == 0:
        print("[inspect] WARNING: empty render (camera not ready)", flush=True)
    else:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        img = Image.fromarray(np.ascontiguousarray(arr))
        img.save(str(out))
        print(f"[inspect] screenshot -> {out}", flush=True)
    print("[inspect] tcp candidates: gripper_end / j_gripper_end", flush=True)
    print("[inspect] DONE", flush=True)
    app.close()
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
