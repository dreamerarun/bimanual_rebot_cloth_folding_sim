"""Isaac Sim scene: table, two WAM arms, a deformable cloth and cameras."""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass

import numpy as np

from .config import ASSET_DIR, CollectCfg, SceneCfg, TABLE_USD


# ---------------------------------------------------------------------------
# app lifecycle
# ---------------------------------------------------------------------------
def make_simulation_app(headless: bool = True, rendering: bool = True):
    """Create the SimulationApp. Must be the very first isaac import."""
    import pathlib

    import isaacsim
    from isaacsim import SimulationApp

    # The default experience loaded by SimulationApp from the pip package is
    # `isaacsim.exp.base.python.kit`, which does *not* include the RTX render
    # delegate: render products come back blank.  Use the full experience.
    exp = pathlib.Path(isaacsim.__file__).parent / "apps" / "isaacsim.exp.full.kit"
    settings = {
        "headless": headless,
        "hide_ui": headless,
        "width": 640,
        "height": 480,
        "renderer": "RayTracedLighting",
        "anti_aliasing": 0,
        "denoiser": False,
        "multi_gpu": False,
        "fast_shutdown": True,
        "create_new_stage": True,
    }
    app = SimulationApp(settings, experience=str(exp) if exp.exists() else "")
    # better default for deformables / cloth
    import carb

    s = carb.settings.get_settings()
    for key, val in {
        "/persist/rtx/reflections/enabled": False,
        "/persist/rtx/transparentReflections/enabled": False,
        "/rtx/shadows/enabled": True,
        "/physics/updateToUsd": True,          # keep USD mesh points in sync (cloth readback)
        "/physics/updateVelocitiesToUsd": True,
        "/physics/fabricEnabled": False,
    }.items():
        try:
            s.set(key, val)
        except Exception:
            pass
    return app


# ---------------------------------------------------------------------------
# helper prim builders
# ---------------------------------------------------------------------------
def _add_lighting(stage):
    from pxr import Gf, UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(700.0)
    dome.CreateColorAttr(Gf.Vec3f(0.95, 0.96, 1.0))

    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(2500.0)
    key.CreateAngleAttr(8.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    from pxr import UsdGeom

    xf = UsdGeom.Xformable(key.GetPrim())
    xf.AddRotateXYZOp().Set(Gf.Vec3f(35.0, 0.0, 25.0))

    fill = UsdLux.DistantLight.Define(stage, "/World/Lights/Fill")
    fill.CreateIntensityAttr(700.0)
    fill.CreateColorAttr(Gf.Vec3f(0.8, 0.88, 1.0))
    UsdGeom.Xformable(fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(55.0, 0.0, 200.0))


def _set_xform(prim, translate=None, rotate_z=None, scale=None):
    from pxr import Gf, UsdGeom

    api = UsdGeom.XformCommonAPI(prim)
    if scale is not None:
        api.SetScale(Gf.Vec3f(*scale))
    if rotate_z is not None:
        api.SetRotate(Gf.Vec3f(0.0, 0.0, math.degrees(rotate_z)))
    if translate is not None:
        api.SetTranslate(Gf.Vec3d(*translate))


def add_table(world, cfg: SceneCfg):
    """Table: an invisible exact collider + the NVIDIA packing-table visual.

    The NVIDIA Isaac 'PackingTable' prop is a fully-authored physics asset,
    but its dimensions change between releases, so we always place our own
    collider with the exact size we control and reference the pretty mesh on
    top (scaled to fit).  If the USD is missing we fall back to a simple box.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    t = cfg.table
    sx, sy, sz = t.size
    top = t.height

    # --- invisible, exact collider ----------------------------------------
    collider_path = "/World/Table/Collider"
    cube = UsdGeom.Cube.Define(stage, collider_path)
    cube.GetSizeAttr().Set(1.0)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _set_xform(cube.GetPrim(), translate=(0.0, 0.0, top - sz / 2.0), scale=(sx, sy, sz))

    # --- legs (visual) ----------------------------------------------------
    leg_h = top - sz
    for i, (lx, ly) in enumerate(
        [(-sx / 2 + 0.08, -sy / 2 + 0.08), (sx / 2 - 0.08, -sy / 2 + 0.08),
         (-sx / 2 + 0.08, sy / 2 - 0.08), (sx / 2 - 0.08, sy / 2 - 0.08)]
    ):
        leg = UsdGeom.Cube.Define(stage, f"/World/Table/Leg_{i}")
        leg.GetSizeAttr().Set(1.0)
        _set_xform(leg.GetPrim(), translate=(lx, ly, leg_h / 2.0), scale=(0.06, 0.06, leg_h))

    mat = _make_material(stage, "/World/Looks/TableWood", (0.45, 0.32, 0.20), roughness=0.55)
    for prim in stage.GetPrimAtPath("/World/Table").GetChildren():
        UsdShade_bind(mat, prim)

    # --- optional NVIDIA visual ------------------------------------------
    if t.use_usd and pathlib.Path(TABLE_USD).exists():
        vis_path = "/World/Table/NvidiaPackingTable"
        vis = UsdGeom.Xform.Define(stage, vis_path)
        vis.GetPrim().GetReferences().AddReference(str(TABLE_USD))
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(vis.GetPrim())
        r = bbox.ComputeAlignedRange()
        mn, mx = np.array(r.GetMin()), np.array(r.GetMax())
        size = mx - mn
        if np.all(size > 1e-3) and np.all(size < 50.0):
            # scale so the table top width matches our collider and the top
            # surface sits at t.height
            s = min(sx / max(size[0], 1e-3), sy / max(size[1], 1e-3))
            center = (mn + mx) / 2.0
            _set_xform(
                vis.GetPrim(),
                translate=(0.0, 0.0, top - s * mx[2]),
                scale=(s, s, s),
            )
            from pxr import UsdGeom as _G

            # shift so bbox centre in x/y is on the origin
            _G.Xformable(vis.GetPrim()).AddTranslateOp(opSuffix="center").Set(Gf.Vec3d(-center[0] * s, -center[1] * s, 0.0))
            # referenced assets must not be baked into our layer
            vis.GetPrim().SetInstanceable(False)
    return stage.GetPrimAtPath("/World/Table/Collider")


def add_pedestal(world, cfg: SceneCfg):
    """A low rigid board under the cloth.

    The cloth overhangs it by a few centimetres, so the gripper finger pads can
    dip below the cloth surface at the exposed edge and actually pinch the
    sheet instead of being blocked by the table top.
    """
    from pxr import UsdGeom, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    c = cfg.cloth
    h = float(getattr(c, "support_height", 0.0))
    if h <= 0.0:
        return None
    sx, sy = getattr(c, "support_size", (0.42, 0.42))
    top = cfg.table.height + h
    prim = UsdGeom.Cube.Define(stage, "/World/Pedestal")
    prim.GetSizeAttr().Set(1.0)
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    _set_xform(prim.GetPrim(), translate=(0.0, 0.0, cfg.table.height + h / 2.0),
               scale=(sx, sy, h))
    mat = _make_material(stage, "/World/Looks/Pedestal", (0.32, 0.24, 0.16), roughness=0.7)
    UsdShade_bind(mat, prim.GetPrim())
    return prim.GetPrim()


def UsdShade_bind(mat, prim):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants)


def _make_material(stage, path, color, roughness=0.5, metallic=0.0):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


# ---------------------------------------------------------------------------
# cloth
# ---------------------------------------------------------------------------
class Cloth:
    """Surface-deformable cloth (PhysX 5 deformable body)."""

    def __init__(self, cfg: SceneCfg, rng=None):
        import omni.usd
        from omni.physx.scripts import deformableUtils
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

        self.cfg = cfg.cloth
        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        nx = ny = c.resolution
        w, h = c.size
        z = cfg.table.height + getattr(c, "support_height", 0.0) + c.height_offset
        cx, cy = c.center
        jit = float(getattr(c, "jitter_xy", 0.0) or 0.0)
        self.pose_offset = (0.0, 0.0)
        if rng is not None and jit > 0.0:
            self.pose_offset = (float(rng.uniform(-jit, jit)), float(rng.uniform(-jit, jit)))
        dx, dy = self.pose_offset
        print(f"[Cloth] build offset dx={dx*100:+.2f} cm dy={dy*100:+.2f} cm")

        self.root_path = "/World/Cloth"
        self.skin_path = self.root_path + "/mesh"
        self.sim_path = self.root_path + "/simMesh"

        root = UsdGeom.Xform.Define(stage, self.root_path)
        mesh = UsdGeom.Mesh.Define(stage, self.skin_path)

        xs = np.linspace(-w / 2, w / 2, nx + 1)
        ys = np.linspace(-h / 2, h / 2, ny + 1)
        points, uvs, idx = [], [], []
        for j in range(ny + 1):
            for i in range(nx + 1):
                points.append(Gf.Vec3f(cx + xs[i] + dx, cy + ys[j] + dy, z))
                uvs.append(Gf.Vec2f(i / nx, j / ny))
        for j in range(ny):
            for i in range(nx):
                a = j * (nx + 1) + i
                b = a + 1
                c2 = a + nx + 1
                d = c2 + 1
                idx += [a, b, d, a, d, c2]
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([3] * (2 * nx * ny))
        mesh.CreateFaceVertexIndicesAttr(idx)
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * len(points))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateDoubleSidedAttr(True)
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        primvar.Set(uvs)

        mat = _make_material(stage, "/World/Looks/Cloth", (0.18, 0.42, 0.75), roughness=0.85)
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

        ok = deformableUtils.create_auto_surface_deformable_hierarchy(
            stage,
            root_prim_path=self.root_path,
            simulation_mesh_path=self.sim_path,
            cooking_src_mesh_path=self.skin_path,
            cooking_src_simplification_enabled=c.subdivision_simplification,
            set_visibility_with_guide_purpose=True,
        )
        if not ok:
            raise RuntimeError("failed to create the deformable cloth")

        body = root.GetPrim()
        body.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
        body.GetAttribute("physxDeformableBody:selfCollision").Set(False)
        body.GetAttribute("physxDeformableBody:enableSpeculativeCCD").Set(True)
        body.GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(12)
        body.GetAttribute("physxDeformableBody:maxLinearVelocity").Set(8.0)
        if body.HasAttribute("physxDeformableBody:mass"):
            body.GetAttribute("physxDeformableBody:mass").Set(c.mass)

        sim_prim = stage.GetPrimAtPath(self.sim_path)
        sim_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
        PhysxSchema.PhysxCollisionAPI(sim_prim).GetRestOffsetAttr().Set(c.thickness * 0.5)
        PhysxSchema.PhysxCollisionAPI(sim_prim).GetContactOffsetAttr().Set(c.thickness * 1.5)

        mat_path = "/World/Looks/ClothDeformable"
        deformableUtils.add_surface_deformable_material(
            stage,
            mat_path,
            dynamic_friction=c.dynamic_friction,
            surface_thickness=c.thickness,
            surface_stretch_stiffness=c.stretch_stiffness,
            surface_shear_stiffness=c.shear_stiffness,
            surface_bend_stiffness=c.bend_stiffness,
        )
        from omni.physx.scripts import physicsUtils

        physicsUtils.add_physics_material_to_prim(stage, body, mat_path)
        dm = stage.GetPrimAtPath(mat_path)
        dm.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
        dm.GetAttribute("physxDeformableMaterial:bendDamping").Set(0.05)
        dm.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.1)

        self.root = root
        self.mesh = mesh

    # -- state -------------------------------------------------------------
    def get_points(self) -> np.ndarray:
        from pxr import UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.sim_path)
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        return np.array([[p[0], p[1], p[2]] for p in pts])

    def get_skin_points(self) -> np.ndarray:
        from pxr import UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.skin_path)
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        return np.array([[p[0], p[1], p[2]] for p in pts])

    def set_points(self, pts: np.ndarray):
        """Used to re-roll the cloth to its initial flat state on reset."""
        from pxr import Gf, UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        for path in (self.sim_path, self.skin_path):
            m = UsdGeom.Mesh(stage.GetPrimAtPath(path))
            attr = m.GetPointsAttr()
            cur = attr.Get()
            if cur is not None and len(cur) != len(pts):
                continue
            attr.Set([Gf.Vec3f(*p) for p in pts])

    def snapshot(self):
        """Remember the initial flat state so it can be restored between episodes."""
        try:
            self._snap_sim = self.get_points()
            self._snap_skin = self.get_skin_points()
        except Exception as exc:
            self._snap_sim = None
            self._snap_skin = None
            print(f"[Cloth] snapshot note: {exc}")

    def restore(self):
        """Re-roll the cloth to its initial flat state (best effort).

        The deformable sim-mesh points are authored in world coordinates and
        PhysX writes the simulated positions back to them each frame, so
        writing the remembered flat points resets the sheet between episodes
        without resetting the whole physics timeline.
        """
        from pxr import Gf, UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        for path, snap in ((self.sim_path, getattr(self, "_snap_sim", None)),
                           (self.skin_path, getattr(self, "_snap_skin", None))):
            if snap is None:
                continue
            m = UsdGeom.Mesh(stage.GetPrimAtPath(path))
            attr = m.GetPointsAttr()
            cur = attr.Get()
            if cur is not None and len(cur) != len(snap):
                continue
            attr.Set([Gf.Vec3f(*p) for p in snap])

    def reset(self):
        self.restore()
