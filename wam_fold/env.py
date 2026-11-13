"""Two-arm WAM cloth-folding environment built directly on Isaac Sim 5.1."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import SceneCfg
from .kinematics import ArmKinematics, axis_angle_to_matrix


def _to_np(x):
    """Works whether the backend returns numpy, torch or warp arrays."""
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera orientation (w,x,y,z) looking from eye at target."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    up = np.asarray(up, dtype=float)
    # camera looks along -Z in USD
    z = -f
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    # rotation matrix -> quaternion (w,x,y,z)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([w, qx, qy, qz])


class WamClothEnv:
    """Headless / GUI-able environment. One env per process (Isaac Sim limitation)."""

    N_ARM_JOINTS = 7
    ACTION_DIM = 16

    def __init__(self, cfg: SceneCfg | None = None, headless: bool = True, cameras: bool = True, seed: int = 0,
                 cloth_readback: bool = False):
        self.cfg = cfg or SceneCfg()
        self.headless = headless
        self.use_cameras = cameras
        self.cloth_readback = cloth_readback
        self.rng = np.random.default_rng(seed)

        from .scene import make_simulation_app

        self.app = make_simulation_app(headless=headless, rendering=cameras)
        self._build()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build(self):
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        from .scene import Cloth, _add_lighting, add_table, _set_xform

        self._ArticulationAction = ArticulationAction
        stage = omni.usd.get_context().get_stage()
        self.stage = stage

        # GPU physics + fabric is required for deformable (cloth) bodies.
        try:
            from isaacsim.core.simulation_manager import SimulationManager

            SimulationManager.set_physics_sim_device("cuda:0")
            # The CUDA pipeline needs the torch backend.  Pin it *before* any
            # articulation wrapper is constructed, otherwise those wrappers
            # cache numpy backend utils and later reject cuda tensors.
            SimulationManager.set_backend("torch")
            if self.cloth_readback:
                # Debug/eval mode: keep USD mesh points in sync with PhysX so
                # we can measure the cloth.  This costs performance.
                import carb

                carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
                SimulationManager.enable_fabric(False)
        except Exception as exc:  # pragma: no cover
            print("[env] gpu device warning:", exc)

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.cfg.physics_dt,
            rendering_dt=self.cfg.rendering_dt,
            backend="torch",
        )

        # physics scene tweaks (the World above already created /physicsScene)
        try:
            from pxr import PhysxSchema

            scene_prim = stage.GetPrimAtPath("/physicsScene")
            if not scene_prim:
                scene_prim = UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim()
            scene_prim.ApplyAPI(PhysxSchema.PhysxSceneAPI)
            api = PhysxSchema.PhysxSceneAPI(scene_prim)
            api.GetTimeStepsPerSecondAttr().Set(round(1.0 / self.cfg.physics_dt))
            api.GetGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1024 * 1024)
            api.GetEnableGPUDynamicsAttr().Set(True)
        except Exception as exc:  # pragma: no cover
            print("[env] physx scene setup warning:", exc)

        # large static ground (local prim, no remote asset needed)
        ground = UsdGeom.Cube.Define(stage, "/World/Ground")
        ground.GetSizeAttr().Set(1.0)
        _set_xform(ground.GetPrim(), translate=(0.0, 0.0, -0.03), scale=(8.0, 8.0, 0.04))
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

        _add_lighting(stage)
        add_table(self.world, self.cfg)

        # GPU dynamics is mandatory for deformable bodies; must be on before
        # any deformable schema is authored.
        try:
            from pxr import PhysxSchema
            from isaacsim.core.simulation_manager import SimulationManager

            scene_prim = stage.GetPrimAtPath("/physicsScene")
            scene_prim.ApplyAPI(PhysxSchema.PhysxSceneAPI)
            api = PhysxSchema.PhysxSceneAPI(scene_prim)
            if not api.GetEnableGPUDynamicsAttr():
                api.CreateEnableGPUDynamicsAttr(True)
            api.GetEnableGPUDynamicsAttr().Set(True)
            SimulationManager.enable_gpu_dynamics(True, physics_scene="/physicsScene")
            self.world.get_physics_context().set_gpu_max_soft_body_contacts(4 * 1024 * 1024)
        except Exception as exc:  # pragma: no cover
            print("[env] gpu dynamics warning:", exc)

        # The CUDA physics pipeline forces the torch backend, so every array we
        # hand to the articulation API has to be a torch tensor on that device.
        self._torch_device = None
        try:
            from isaacsim.core.simulation_manager import SimulationManager

            if SimulationManager.get_backend() == "torch":
                import carb

                device_id = carb.settings.get_settings().get_as_int("/physics/cudaDevice")
                self._torch_device = f"cuda:{device_id}" if device_id >= 0 else "cpu"
        except Exception:
            pass

        # --- robots -----------------------------------------------------
        rc = self.cfg.robot
        self.names = ("left", "right")
        self.robots = {}
        self.kin = {}
        bases = {
            "left": (rc.left_base_xyz, rc.left_base_yaw),
            "right": (rc.right_base_xyz, rc.right_base_yaw),
        }
        for side in self.names:
            base_xyz, yaw = bases[side]
            prim_path = f"/World/Robots/WAM_{side}"
            root_prim = UsdGeom.Xform.Define(stage, prim_path)
            _set_xform(root_prim.GetPrim(), translate=base_xyz, rotate_z=yaw)
            robot_prim_path = prim_path + "/robot"
            prim = UsdGeom.Xform.Define(stage, robot_prim_path)
            prim.GetPrim().GetReferences().AddReference(rc.usd_path)

            robot = Robot(prim_path=robot_prim_path, name=f"wam_{side}")
            self.world.scene.add(robot)
            self.robots[side] = robot

            k = ArmKinematics(rc.urdf_path, rc.tcp_link)
            k.base_pos = np.array(base_xyz, dtype=float)
            k.base_yaw = yaw
            self.kin[side] = k

        # --- cloth ------------------------------------------------------
        self.cloth = Cloth(self.cfg)

        self.world.reset()

        # cache joint indices (planar/finger joints are interleaved)
        self.dof_names = list(self.robots["left"].dof_names)
        self.arm_idx, self.finger_idx = [], []
        for i, n in enumerate(self.dof_names):
            if any(f in n for f in ("finger", "bhand")):
                self.finger_idx.append(i)
            else:
                self.arm_idx.append(i)
        self.arm_idx = np.array(self.arm_idx)
        self.finger_idx = np.array(self.finger_idx)
        assert len(self.arm_idx) == 7, self.dof_names

        # per-finger-joint open / closed targets, matched by joint name
        curl = {k: (o, c) for k, o, c in rc.finger_curl}
        self.finger_open_q = np.zeros(len(self.finger_idx))
        self.finger_closed_q = np.zeros(len(self.finger_idx))
        for i, idx in enumerate(self.finger_idx):
            name = self.dof_names[idx]
            for key, (o, c) in curl.items():
                if key in name:
                    self.finger_open_q[i] = o
                    self.finger_closed_q[i] = c
                    break

        # --- cameras ----------------------------------------------------
        self.cameras = {}
        if self.use_cameras:
            self._build_cameras()

        self._n_substeps = max(1, int(round((1.0 / self.cfg.control_hz) / self.cfg.physics_dt)))
        self.reset(render=self.use_cameras)
        self._last_action = self.get_state()

    # ------------------------------------------------------------------
    def _arr(self, x):
        """Convert an array to whatever backend the articulation expects."""
        if self._torch_device is not None:
            import torch

            return torch.tensor(np.asarray(x, dtype=np.float32), device=self._torch_device)
        return np.asarray(x, dtype=np.float32)

    # ------------------------------------------------------------------
    def _build_cameras(self):
        from isaacsim.sensors.camera import Camera

        for c in self.cfg.cameras:
            cam = Camera(
                prim_path=f"/World/Cameras/{c.name}",
                resolution=c.resolution,
                frequency=30,
            )
            cam.initialize()
            cam.set_world_pose(
                position=np.array(c.position, dtype=np.float32),
                orientation=look_at_quat(c.position, c.look_at),
                camera_axes="usd",
            )
            try:
                # Camera.set_focal_length() wants tenths of a stage unit, which
                # is ambiguous; author the USD attribute (millimetres) directly.
                from pxr import UsdGeom

                UsdGeom.Camera(cam.prim).GetFocalLengthAttr().Set(float(c.focal_length))
            except Exception:
                pass
            # USD cameras default to a 1 m near clip which would clip the
            # whole table-top scene (camera is only ~0.9 m away).
            try:
                cam.set_clipping_range(0.01, 100.0)
            except Exception:
                from pxr import Gf, UsdGeom

                UsdGeom.Camera(cam.prim).GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
            self.cameras[c.name] = cam
        # first frame
        self.world.step(render=True)

    # ------------------------------------------------------------------
    # state / action
    # ------------------------------------------------------------------
    def get_arm_q(self, side) -> np.ndarray:
        q = _to_np(self.robots[side].get_joint_positions())
        return q[self.arm_idx]

    def get_gripper(self, side) -> float:
        q = _to_np(self.robots[side].get_joint_positions())[self.finger_idx]
        span = self.finger_closed_q - self.finger_open_q
        return float(np.clip(np.mean((q - self.finger_open_q) / (span + 1e-8)), 0, 1))

    def get_state(self) -> np.ndarray:
        s = []
        for side in self.names:
            s.append(self.get_arm_q(side))
            s.append([self.get_gripper(side)])
        return np.concatenate(s).astype(np.float32)

    def apply_action(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape[0] == self.ACTION_DIM, action.shape
        rc = self.cfg.robot
        for k, side in enumerate(self.names):
            a = action[k * 8 : (k + 1) * 8]
            kk = self.kin[side]
            q_arm = np.clip(a[:7], kk.lower + 1e-3, kk.upper - 1e-3)
            grip = float(np.clip(a[7], 0.0, 1.0))
            q = np.zeros(len(self.dof_names))
            q[self.arm_idx] = q_arm
            q[self.finger_idx] = self.finger_open_q + grip * (self.finger_closed_q - self.finger_open_q)
            self.robots[side].apply_action(self._ArticulationAction(joint_positions=self._arr(q)))
        self._last_action = action

    def step(self, action=None, render: bool | None = None):
        if action is not None:
            self.apply_action(action)
        render = self.use_cameras if render is None else render
        for i in range(self._n_substeps):
            self.world.step(render=render and (i == self._n_substeps - 1))

    # ------------------------------------------------------------------
    def get_images(self) -> dict:
        out = {}
        for name, cam in self.cameras.items():
            rgba = cam.get_rgba()
            if rgba is None:
                continue
            out[name] = np.asarray(rgba)[..., :3].astype(np.uint8)
        return out

    def get_obs(self) -> dict:
        obs = {"state": self.get_state(), "joint_q": {s: self.get_arm_q(s) for s in self.names}}
        if self.use_cameras:
            obs["images"] = self.get_images()
        return obs

    # ------------------------------------------------------------------
    # kinematics helpers (targets are in world frame)
    # ------------------------------------------------------------------
    def ik(self, side: str, pos_world, rot_world=None, q0=None, **kw):
        k = self.kin[side]
        R0 = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), k.base_yaw)
        p_local = R0.T @ (np.asarray(pos_world, dtype=float) - k.base_pos)
        R_local = None if rot_world is None else R0.T @ rot_world
        return k.solve(p_local, R_local, q0=q0, **kw)

    def tcp_pose(self, side: str):
        """TCP position in world frame from the *simulated* joint angles.

        USD transforms are not updated when the fabric is enabled, so we use
        forward kinematics on the joint state that PhysX reports instead.
        """
        k = self.kin[side]
        p_local, _ = k.fk(self.get_arm_q(side))
        R0 = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), k.base_yaw)
        return k.base_pos + R0 @ p_local

    def tcp_rot(self, side: str):
        """TCP rotation in world frame (columns: tool x, y, z)."""
        k = self.kin[side]
        _, R_local = k.fk(self.get_arm_q(side))
        R0 = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), k.base_yaw)
        return R0 @ R_local

    def obb_rotation(self, side, yaw=0.0):
        k = self.kin[side]
        # top-down grasp: tool Z points down, tool X points along the arm's yaw
        Rz = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), yaw)
        Rx = axis_angle_to_matrix(np.array([1.0, 0.0, 0.0]), math.pi)
        return Rz @ Rx

    def reset(self, render: bool | None = None):
        render = self.use_cameras if render is None else render
        self.world.reset()
        for side in self.names:
            q0 = self.kin[side].home_q
            q = np.zeros(len(self.dof_names))
            q[self.arm_idx] = q0
            q[self.finger_idx] = self.finger_open_q
            self.robots[side].set_joint_positions(self._arr(q))
        self.world.step(render=render)

    def close(self):
        try:
            self.app.close()
        except Exception:
            pass
