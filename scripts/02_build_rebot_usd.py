#!/usr/bin/env python
"""Build a USD asset for the Seeed reBot Arm T701 HT (B601) from its URDF.

Same route that already worked for the Barrett WAM: Isaac Sim's URDF importer
(`URDFParseAndImportFile` with an `ImportConfig`), i.e. NOT Seeed's payload-based
USD.  We use it because the payload-USD route produced an articulation that Isaac
Sim 5.1 refuses to initialise (0 DOFs, joint-body resolution failures).

Run (headless):
    env -u PYTHONPATH -u LD_LIBRARY_PATH \
      ~/miniconda3/envs/leisaac/bin/python scripts/02_build_rebot_usd.py

Output:
    assets/rebot_t701/usd/rebot_b601_flat.usd
"""
from __future__ import annotations

import argparse
import pathlib
import sys

PROJECT = pathlib.Path(__file__).resolve().parents[1]

URDF = PROJECT / "assets" / "rebot_t701" / "urdf" / "RS" / "urdf" / "ReBot_Arm_RS.urdf"
OUT_DIR = PROJECT / "assets" / "rebot_t701" / "usd"
OUT = OUT_DIR / "rebot_b601_flat.usd"


def convert(urdf: pathlib.Path, usd_path: pathlib.Path) -> None:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "hide_ui": True})
    try:
        import omni.kit.commands
        import omni.usd

        omni.usd.get_context().new_stage()
        ok, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
        assert ok
        cfg.set_merge_fixed_joints(False)   # keep gripper fixed joints explicit
        cfg.set_fix_base(True)
        cfg.set_self_collision(False)
        cfg.set_convex_decomp(True)         # convex hulls for fingers
        cfg.set_collision_from_visuals(False)
        cfg.set_default_drive_type(1)       # position drives
        cfg.set_default_drive_strength(1.0e4)
        cfg.set_default_position_drive_damping(1.0e3)
        cfg.set_import_inertia_tensor(True)
        cfg.set_distance_scale(1.0)
        cfg.set_make_default_prim(False)

        usd_path.parent.mkdir(parents=True, exist_ok=True)
        ok, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(urdf),
            import_config=cfg,
            dest_path=str(usd_path),
            get_articulation_root=True,
        )
        print(f"[build] imported -> {prim_path}")
        stage = omni.usd.get_context().get_stage()
        stage.Save()

        from pxr import Usd, UsdPhysics

        # NOTE: Do NOT apply ArticulationRootAPI manually here.
        # Isaac Sim 5.1 automatically adds PhysicsArticulationRootAPI when the
        # USD is referenced into a stage with PhysX active.  Manual application
        # causes "Nested articulation roots are not allowed".
        stage2 = Usd.Stage.Open(str(usd_path))
        joints = [p for p in stage2.Traverse() if p.GetTypeName() == "PhysicsRevoluteJoint"]
        print(f"[build] revolute joints: {len(joints)}")
        for p in joints:
            ax = UsdPhysics.RevoluteJoint(p).GetAxisAttr().Get()
            print(f"        {p.GetPath()} axis={ax}")
        prims = [p for p in stage2.Traverse() if p.GetTypeName() == "PhysicsPrismaticJoint"]
        print(f"[build] prismatic joints: {len(prims)}")
        for p in prims:
            print(f"        {p.GetPath()}")
        root = stage2.GetPrimAtPath("/ReBot_Arm_RS")
        print(f"[build] /ReBot_Arm_RS valid: {bool(root)}")
    finally:
        app.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=str(URDF))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    convert(pathlib.Path(args.urdf), pathlib.Path(args.out))
    print(f"[build] done: {args.out}")


if __name__ == "__main__":
    main()
