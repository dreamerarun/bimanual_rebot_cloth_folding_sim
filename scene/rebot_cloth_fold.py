#!/usr/bin/env python
"""reBot Arm T701 HT (B601) bimanual + deformable cloth fold scene (Isaac Sim 5.1).

Standalone Isaac Sim script: ground + table + PhysX surface-deformable cloth +
TWO reBot arms mounted on opposite table ends + top/front cameras.

Run:
    env -u PYTHONPATH -u LD_LIBRARY_PATH \
      ~/miniconda3/envs/leisaac/bin/python scene/rebot_cloth_fold.py --task single_fold --gui

    env -u PYTHONPATH -u LD_LIBRARY_PATH \
      ~/miniconda3/envs/leisaac/bin/python scene/rebot_cloth_fold.py --task single_fold
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

PROJECT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from wam_fold.config import SceneCfg  # noqa: E402
from wam_fold.env import look_at_quat  # noqa: E402

# USD built by scripts/02_build_rebot_usd.py from the Seeed URDF.
# NOTE: the URDF importer does not set a defaultPrim, so we reference the
# explicit root prim path "/ReBot_Arm_RS" (see scripts/02_build_rebot_usd.py).
DEFAULT_ASSET = str(PROJECT / "assets" / "rebot_t701" / "usd" / "rebot_b601_flat.usd")
REFERENCE_PRIM = "/ReBot_Arm_RS"
SIDES = ("left", "right")


def _find_articulation_root(stage, mount_path: str):
    """First prim under the mount Xform carrying ArticulationRootAPI."""
    from pxr import UsdPhysics

    for p in stage.Traverse():
        if str(p.GetPath()).startswith(mount_path) and p.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(p.GetPath())
    return None


def build_scene(cfg: SceneCfg, asset: str, base_x: float, base_z: float):
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": not cfg.headless, "hide_ui": cfg.headless, "width": 640, "height": 480})

    from wam_fold.scene import Cloth, _add_lighting, _set_xform, add_table

    from isaacsim.core.api import World
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdGeom, UsdPhysics

    SimulationManager.set_physics_sim_device("cuda:0")
    SimulationManager.set_backend("torch")
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=cfg.physics_dt,
        rendering_dt=cfg.rendering_dt,
        backend="torch",
    )
    stage = get_current_stage()

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.GetSizeAttr().Set(1.0)
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    _set_xform(ground.GetPrim(), translate=(0.0, 0.0, -0.03), scale=(8.0, 8.0, 0.04))
    _add_lighting(stage)
    add_table(world, cfg)

    # ---- bimanual arms: left at (-base_x, 0) yaw 0, right at (+base_x, 0) yaw pi
    robots = {}
    roots = {}
    bases = {
        "left": ((-base_x, 0.0, base_z), 0.0),
        "right": ((base_x, 0.0, base_z), float(np.pi)),
    }
    for side in SIDES:
        base_xyz, yaw = bases[side]
        mount = f"/World/Robots/reBotArm_{side}"
        parent = UsdGeom.Xform.Define(stage, mount)
        _set_xform(parent.GetPrim(), translate=base_xyz, rotate_z=yaw)
        inner = UsdGeom.Xform.Define(stage, mount + "/robot")
        inner.GetPrim().GetReferences().AddReference(asset, REFERENCE_PRIM)

        root = _find_articulation_root(stage, mount)
        if root is None:
            raise RuntimeError(f"no ArticulationRootAPI under {mount}")
        roots[side] = root
        robot = Robot(prim_path=root, name=f"rebot_{side}")
        world.scene.add(robot)
        robots[side] = robot
        print(f"[scene] {side} arm base={base_xyz} yaw={yaw:.2f} root={root}")

    world.initialize_physics()
    world.step()
    for side in SIDES:
        robots[side].initialize()
        print(f"[scene] {side} dofs: {list(robots[side].dof_names)}")
    world.step()

    cloth = Cloth(cfg)

    cameras = {}
    for c in cfg.cameras:
        from isaacsim.sensors.camera import Camera

        cam = Camera(prim_path=f"/World/Cameras/{c.name}", resolution=c.resolution, frequency=30)
        cam.initialize()
        cam.set_world_pose(
            position=np.array(c.position, dtype=np.float32),
            orientation=look_at_quat(c.position, c.look_at),
            camera_axes="usd",
        )
        try:
            from pxr import UsdGeom as _G

            _G.Camera(cam.prim).GetFocalLengthAttr().Set(float(c.focal_length))
        except Exception:
            pass
        try:
            cam.set_clipping_range(0.01, 100.0)
        except Exception:
            from pxr import Gf, UsdGeom as _G

            _G.Camera(cam.prim).GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
        cameras[c.name] = cam
        print(f"[scene] camera {c.name} at {c.position} look_at {c.look_at}")

    world.step(render=True)
    for _ in range(5):
        world.step(render=True)
    return app, world, cameras, robots, roots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=DEFAULT_ASSET)
    ap.add_argument("--task", default="single_fold", choices=("single_fold", "double_fold"))
    ap.add_argument("--base-x", type=float, default=0.80, help="arm base distance from table centre (m)")
    ap.add_argument("--base-z", type=float, default=0.75, help="arm base height (m), table top height")
    ap.add_argument("--cloth-size", default="0.62,0.62")
    ap.add_argument("--cloth-res", type=int, default=24)
    ap.add_argument("--out", default=str(PROJECT / "outputs" / "rebot_scene.png"))
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    cfg = SceneCfg()
    cfg.headless = not args.gui
    cfg.cloth.resolution = args.cloth_res
    cfg.cloth.size = tuple(float(x) for x in args.cloth_size.split(","))

    app, world, cameras, robots, roots = build_scene(cfg, args.asset, args.base_x, args.base_z)

    from PIL import Image

    for name, cam in cameras.items():
        rgba = cam.get_rgba()
        arr = None if rgba is None else np.asarray(rgba)[..., :3].astype(np.uint8)
        print(f"[scene] cam {name} rgba shape: {None if arr is None else arr.shape}")
        if arr is None or arr.size == 0:
            print(f"[scene] cam {name}: EMPTY RENDER, skipping save")
            continue
        out = pathlib.Path(args.out).with_name(f"rebot_scene_{name}.png")
        Image.fromarray(np.ascontiguousarray(arr)).save(str(out))
        print(f"[scene] saved {out}")

    print(f"[scene] task={args.task} robots={list(robots)} roots={roots}")
    print("[scene] DONE")
    app.close()
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
