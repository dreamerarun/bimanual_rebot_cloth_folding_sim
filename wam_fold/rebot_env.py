from __future__ import annotations

import math
import os
import pathlib
from dataclasses import dataclass, field

import numpy as np

from .config import SceneCfg, RobotCfg, ClothCfg
from .kinematics import ArmKinematics, axis_angle_to_matrix


# --- reBot-specific overrides (can be passed as a RobotCfg subclass or plain dict) ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REBOT_USD   = str(PROJECT_ROOT / "assets" / "rebot_t701" / "usd" / "rebot_b601_flat.usd")
REBOT_URDF  = str(PROJECT_ROOT / "assets" / "rebot_t701" / "urdf" / "RS" / "urdf" / "ReBot_Arm_RS.urdf")
REBOT_TCP   = "gripper_end"


@dataclass
class RebotRobotCfg(RobotCfg):
    """RobotCfg patched for the reBot B601."""
    usd_path:        str   = REBOT_USD
    urdf_path:       str   = REBOT_URDF
    tcp_link:        str   = REBOT_TCP
    # reBot arms sit at x = ±0.55 m on the table (0.75 m) — close enough that
    # the 75°-tilted grasp reaches the whole cloth fold path.
    left_base_xyz:   tuple = (-0.55, 0.0, 0.75)
    right_base_xyz:  tuple = ( 0.55, 0.0, 0.75)
    left_base_yaw:   float = 0.0
    right_base_yaw:  float = float(math.pi)
    stiffness:       float = 3.0e4
    damping:         float = 3.0e3
    # prismatic finger stroke: open = 0.045 m, closed = 0.0 m (per finger)
    finger_open:     float = 0.045
    finger_closed:   float = 0.0
    # finger joint names expected in the URDF / USD
    finger_joint_names: tuple = ("gripper_joint1", "gripper_joint2")
    n_arm_joints:    int   = 6
    ROBOT_ROOT_PATH: str   = "/World/Robots/reBotArm_%s/robot/root_joint"


def _to_np(x):
    """Works whether the backend returns numpy, torch, or warp arrays."""
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera orientation quaternion (w,x,y,z) looking from eye at target."""
    from .env import look_at_quat as _laq
    return _laq(eye, target, up)


class RebotClothEnv:
    """Isaac Sim bimanual cloth-folding environment for the reBot B601.

    One instance per process (Isaac Sim limitation).
    """

    N_ARM_JOINTS = 6
    ACTION_DIM   = (N_ARM_JOINTS + 1) * 2   # = 14
    REFERENCE_PRIM = "/ReBot_Arm_RS"         # defaultPrim in the converted USD
    ROBOT_ROOT_PATH = "/World/Robots/reBotArm_%s/robot/root_joint"


    def __init__(
        self,
        cfg:              SceneCfg | None = None,
        robot_cfg:        RebotRobotCfg | None = None,
        headless:         bool = True,
        cameras:          bool = True,
        seed:             int  = 0,
        cloth_readback:   bool = False,
        hide_ui:          bool | None = None,
        window_size:      tuple | None = None,
    ):
        self.cfg        = cfg or SceneCfg()
        self.rcfg       = robot_cfg or RebotRobotCfg()
        # patch SceneCfg.robot so shared helpers still work
        self.cfg.robot  = self.rcfg
        self.headless   = headless
        self.use_cameras = cameras
        self.cloth_readback = cloth_readback
        self.rng        = np.random.default_rng(seed)
        # fold-metric reference (captured from the flat cloth)
        self._cloth_ref_y = None
        self._cloth_near_mask = None
        self._cloth_ref_center = 0.0
        self._cloth_ref_half = 0.13
        self._cloth_near_start = -0.13
        self._cloth_near_target = 0.13

        # Create SimulationApp directly (same as the working scene script,
        # NOT using make_simulation_app which sets extra carb settings that
        # cause hangs with the reBot articulation).
        from isaacsim import SimulationApp
        win_w, win_h = window_size or ((640, 480) if headless else (960, 540))
        self.app = SimulationApp({
            "headless": headless,
            "hide_ui": headless if hide_ui is None else hide_ui,
            "width": win_w,
            "height": win_h,
        })

        # Keep PhysX -> USD in sync so cloth vertices can be read back for the
        # fold-success metric.  Must be set before the World / cloth is built.
        try:
            import carb
            cs = carb.settings.get_settings()
            cs.set_bool("/physics/updateToUsd", True)
            cs.set_bool("/physics/updateVelocitiesToUsd", True)
            if cloth_readback:
                cs.set_bool("/physics/suppressReadback", False)
        except Exception as exc:
            print(f"[RebotClothEnv] carb readback settings note: {exc}")

        self._build()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build(self):
        from isaacsim.core.api import World
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom, UsdPhysics

        from .scene import Cloth, _add_lighting, add_pedestal, add_table, _set_xform

        self._ArticulationAction = ArticulationAction

        # GPU physics + torch backend — must be set BEFORE World creation
        # (same order as the working scene/rebot_cloth_fold.py)
        try:
            from isaacsim.core.simulation_manager import SimulationManager
            SimulationManager.set_physics_sim_device("cuda:0")
            SimulationManager.set_backend("torch")
            if self.cloth_readback:
                import carb
                # set_physics_sim_device() force-enables fabric; disable it
                # again so PhysX writes the deformable sim-mesh points back to
                # USD (required for get_cloth_points / success metrics).
                carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
                carb.settings.get_settings().set_bool("/physics/fabricEnabled", False)
                SimulationManager.enable_fabric(False)
        except Exception as exc:
            print(f"[RebotClothEnv] gpu device note: {exc}")

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.cfg.physics_dt,
            rendering_dt=self.cfg.rendering_dt,
            backend="torch",
        )
        stage = get_current_stage()
        self.stage = stage

        # Enable GPU dynamics on the physics scene *prim* (deformable bodies
        # require it) before the cloth is authored.  Do NOT hand-define
        # /physicsScene — let SimulationManager create/configure it, otherwise
        # its CCD handling silently disables GPU deformables.
        try:
            from isaacsim.core.simulation_manager import SimulationManager

            # Make sure /physicsScene exists first (the physics context creates
            # it with the correct defaults), then force the GPU broadphase.
            _phys_ctx = self.world.get_physics_context()
            SimulationManager.set_broadphase_type("GPU", physics_scene="/physicsScene")
            SimulationManager.enable_gpu_dynamics(True, physics_scene="/physicsScene")
            _phys_ctx.set_gpu_max_soft_body_contacts(1024 * 1024)
            try:
                from pxr import PhysxSchema
                scene_prim = stage.GetPrimAtPath("/physicsScene")
                if scene_prim:
                    api = PhysxSchema.PhysxSceneAPI(scene_prim)
                    if api.GetEnableGPUDynamicsAttr():
                        api.GetEnableGPUDynamicsAttr().Set(True)
                    if hasattr(api, "GetGpuMaxDeformableSurfaceContactsAttr"):
                        api.GetGpuMaxDeformableSurfaceContactsAttr().Set(1024 * 1024)
            except Exception:
                pass
            print("[RebotClothEnv] GPU dynamics enabled on /physicsScene "
                  f"(broadphase={SimulationManager.get_broadphase_type()})")
        except Exception as exc:
            print(f"[RebotClothEnv] GPU dynamics context note: {exc}")

        # static ground
        ground = UsdGeom.Cube.Define(stage, "/World/Ground")
        ground.GetSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        _set_xform(ground.GetPrim(), translate=(0.0, 0.0, -0.03), scale=(8.0, 8.0, 0.04))

        _add_lighting(stage)
        add_table(self.world, self.cfg)
        add_pedestal(self.world, self.cfg)

        # --- robots (bimanual) -------------------------------------------
        rc = self.rcfg
        self.names  = ("left", "right")
        self.robots  = {}
        self.kin     = {}
        bases = {
            "left":  (rc.left_base_xyz,  rc.left_base_yaw),
            "right": (rc.right_base_xyz, rc.right_base_yaw),
        }
        for side in self.names:
            base_xyz, yaw = bases[side]
            mount = f"/World/Robots/reBotArm_{side}"
            parent = UsdGeom.Xform.Define(stage, mount)
            _set_xform(parent.GetPrim(), translate=base_xyz, rotate_z=yaw)
            inner = UsdGeom.Xform.Define(stage, mount + "/robot")
            # Must pass the defaultPrim ("/ReBot_Arm_RS") — without it the
            # prim tree doesn't compose and _find_articulation_root returns None
            inner.GetPrim().GetReferences().AddReference(rc.usd_path, self.REFERENCE_PRIM)

            root_path = self._find_articulation_root(stage, mount)
            if root_path is None:
                raise RuntimeError(f"[RebotClothEnv] no ArticulationRootAPI under {mount}")

            robot = Robot(prim_path=root_path, name=f"rebot_{side}")
            self.world.scene.add(robot)
            self.robots[side] = robot
            print(f"[RebotClothEnv] {side} base={base_xyz} yaw={yaw:.2f} root={root_path}")

            k = ArmKinematics(rc.urdf_path, rc.tcp_link)
            k.base_pos = np.array(base_xyz, dtype=float)
            k.base_yaw = float(yaw)
            self.kin[side] = k

        # --- initialise physics and robots --------------------------------
        self.world.initialize_physics()
        self.world.step()
        for side in self.names:
            self.robots[side].initialize()
        self.world.step()

        # --- drive gains (smooth, damped position tracking) ---------------
        # Damped PD gains keep the arms from chattering while still tracking
        # the planned joint trajectory accurately.
        self._apply_drive_gains()
        self.world.step()

        # --- DOF index mapping -------------------------------------------
        self.dof_names   = list(self.robots["left"].dof_names)
        self.arm_idx     = []
        self.finger_idx  = []
        for i, name in enumerate(self.dof_names):
            if any(fn in name for fn in rc.finger_joint_names):
                self.finger_idx.append(i)
            else:
                self.arm_idx.append(i)
        self.arm_idx    = np.array(self.arm_idx,    dtype=int)
        self.finger_idx = np.array(self.finger_idx, dtype=int)

        if len(self.arm_idx) != rc.n_arm_joints:
            raise AssertionError(
                f"expected {rc.n_arm_joints} arm joints, got {len(self.arm_idx)}. "
                f"dof_names={self.dof_names}"
            )
        print(f"[RebotClothEnv] arm_idx={self.arm_idx.tolist()} finger_idx={self.finger_idx.tolist()}")

        # --- cloth -------------------------------------------------------
        self.cloth = Cloth(self.cfg, rng=self.rng)
        self.cloth.snapshot()

        # --- cameras -----------------------------------------------------
        self.cameras = {}
        self.record_cam = None
        if self.use_cameras:
            self._build_cameras()
        if not self.headless:
            self._setup_record_view()

        # torch device detection (for ArticulationAction backend)
        self._torch_device = None
        try:
            from isaacsim.core.simulation_manager import SimulationManager
            import carb
            if SimulationManager.get_backend() == "torch":
                did = carb.settings.get_settings().get_as_int("/physics/cudaDevice")
                self._torch_device = f"cuda:{did}" if did >= 0 else "cpu"
        except Exception:
            pass

        self._n_substeps = max(1, int(round((1.0 / self.cfg.control_hz) / self.cfg.physics_dt)))

        # Set home position directly (no world.reset — same as scene script)
        for side in self.names:
            q0 = self.kin[side].home_q
            q  = np.zeros(len(self.dof_names), dtype=np.float32)
            q[self.arm_idx]    = q0
            q[self.finger_idx] = self.rcfg.finger_open
            self.robots[side].set_joint_positions(self._arr(q))
        self.world.step(render=self.use_cameras)
        self._last_action = self.get_state()

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_articulation_root(stage, mount_path: str) -> str | None:
        from pxr import UsdPhysics
        for p in stage.Traverse():
            sp = str(p.GetPath())
            if sp.startswith(mount_path) and (p.HasAPI('ArticulationRootAPI') or p.HasAPI('PhysicsArticulationRootAPI')):
                return str(p.GetPath())
        return None

    # ------------------------------------------------------------------
    # drive gains
    # ------------------------------------------------------------------
    #: Per-DOF stiffness / damping in ``dof_names`` order
    #: [joint1..joint6, gripper_joint1, gripper_joint2].
    ARM_KPS = (3.0e4, 3.0e4, 3.0e4, 1.5e4, 1.5e4, 1.5e4)
    ARM_KDS = (3.0e3, 3.0e3, 3.0e3, 1.5e3, 1.5e3, 1.5e3)
    FINGER_KPS = (5.0e4, 5.0e4)
    FINGER_KDS = (5.0e3, 5.0e3)

    def _apply_drive_gains(self):
        kps = np.array(self.ARM_KPS + self.FINGER_KPS, dtype=np.float32)
        kds = np.array(self.ARM_KDS + self.FINGER_KDS, dtype=np.float32)
        # The controller's torch backend requires torch tensors on the sim device.
        kps_out, kds_out = kps, kds
        try:
            from isaacsim.core.simulation_manager import SimulationManager
            if SimulationManager.get_backend() == "torch":
                import carb
                import torch
                did = carb.settings.get_settings().get_as_int("/physics/cudaDevice")
                dev = f"cuda:{did}" if did >= 0 else "cpu"
                kps_out = torch.tensor(kps, dtype=torch.float32, device=dev)
                kds_out = torch.tensor(kds, dtype=torch.float32, device=dev)
        except Exception as exc:
            print(f"[RebotClothEnv] gain tensor note: {exc}")
        for side in self.names:
            try:
                ctrl = self.robots[side].get_articulation_controller()
                ctrl.set_gains(kps=kps_out, kds=kds_out)
            except Exception as exc:
                print(f"[RebotClothEnv] {side} drive gains note: {exc}")
        print(f"[RebotClothEnv] drive gains set kps={kps.tolist()}")

    # ------------------------------------------------------------------
    def _arr(self, x):
        """Convert array to whatever backend the articulation expects."""
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
                from pxr import UsdGeom
                UsdGeom.Camera(cam.prim).GetFocalLengthAttr().Set(float(c.focal_length))
            except Exception:
                pass
            try:
                cam.set_clipping_range(0.01, 100.0)
            except Exception:
                from pxr import Gf, UsdGeom
                UsdGeom.Camera(cam.prim).GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
            self.cameras[c.name] = cam
        self.world.step(render=True)

    # ------------------------------------------------------------------
    def _setup_record_view(self):
        """Frame the workspace for the GUI and make it the active viewport camera.

        The observation cameras (``self.cameras``) are left untouched so the
        images fed to the policy keep the same framing as the training data.
        """
        if os.environ.get("REBOT_NO_RECORD_VIEW"):
            return
        try:
            from isaacsim.sensors.camera import Camera
            from pxr import UsdGeom, Gf
            from omni.kit.viewport.utility import get_active_viewport

            pos = np.array([1.10, -1.60, 1.55], dtype=np.float32)
            target = np.array([0.0, 0.0, 0.78], dtype=np.float32)
            cam = Camera(
                prim_path="/World/Cameras/record",
                resolution=(960, 540),
                frequency=30,
            )
            cam.initialize()
            cam.set_world_pose(
                position=pos,
                orientation=look_at_quat(pos, target),
                camera_axes="usd",
            )
            prim = UsdGeom.Camera(cam.prim)
            prim.GetFocalLengthAttr().Set(24.0)
            prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
            self.record_cam = cam

            vp = get_active_viewport()
            if vp is not None:
                vp.set_active_camera("/World/Cameras/record")
                print("[RebotClothEnv] GUI viewport camera → /World/Cameras/record")

            # render one frame so the camera's render product is populated
            self.world.step(render=True)
        except Exception as exc:
            print(f"[RebotClothEnv] record view note: {exc}")

    def save_record_preview(self, path: str | pathlib.Path) -> bool:
        """Write the record-camera view to a PNG (for framing checks)."""
        if self.record_cam is None:
            return False
        try:
            from PIL import Image
            rgba = self.record_cam.get_rgba()
            if rgba is None:
                return False
            img = np.asarray(rgba)[..., :3].astype(np.uint8)
            Image.fromarray(img).save(str(path))
            return True
        except Exception as exc:
            print(f"[RebotClothEnv] record preview note: {exc}")
            return False

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------
    def get_arm_q(self, side: str) -> np.ndarray:
        """6-element arm joint positions for one side."""
        q = _to_np(self.robots[side].get_joint_positions())
        return q[self.arm_idx]

    def get_gripper(self, side: str) -> float:
        """Gripper open fraction in [0, 1] (0 = closed, 1 = open)."""
        q  = _to_np(self.robots[side].get_joint_positions())[self.finger_idx]
        rc = self.rcfg
        span = rc.finger_open - rc.finger_closed  # prismatic: open > closed
        return float(np.clip(np.mean((q - rc.finger_closed) / (span + 1e-8)), 0.0, 1.0))

    def get_state(self) -> np.ndarray:
        """Flat 14-dim state: [arm_q(6), grip(1)] 	imes 2."""
        s = []
        for side in self.names:
            s.append(self.get_arm_q(side))           # 6
            s.append(np.array([self.get_gripper(side)]))  # 1
        return np.concatenate(s).astype(np.float32)   # 14

    # ------------------------------------------------------------------
    # action
    # ------------------------------------------------------------------
    def apply_action(self, action: np.ndarray):
        """Apply a 14-dim action [q_arm(6), grip(1)] 	imes 2 to both robots."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape[0] == self.ACTION_DIM,             f"expected {self.ACTION_DIM}, got {action.shape[0]}"
        rc = self.rcfg
        for k, side in enumerate(self.names):
            a = action[k * 7 : (k + 1) * 7]   # 6 arm + 1 grip
            kin = self.kin[side]
            q_arm = np.clip(a[:6], kin.lower + 1e-3, kin.upper - 1e-3)
            grip  = float(np.clip(a[6], 0.0, 1.0))

            # Build full DOF array (arm joints + finger joints)
            q = np.zeros(len(self.dof_names), dtype=np.float32)
            q[self.arm_idx]    = q_arm
            # Both prismatic fingers get the same position command
            finger_pos = rc.finger_closed + grip * (rc.finger_open - rc.finger_closed)
            q[self.finger_idx] = finger_pos

            self.robots[side].apply_action(
                self._ArticulationAction(joint_positions=self._arr(q))
            )
        self._last_action = action

    def step(self, action=None, render: bool | None = None):
        """Advance the simulation by one control tick (1/20 s)."""
        if action is not None:
            self.apply_action(action)
        render = self.use_cameras if render is None else render
        for i in range(self._n_substeps):
            self.world.step(render=render and (i == self._n_substeps - 1))

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def get_images(self) -> dict[str, np.ndarray]:
        out = {}
        for name, cam in self.cameras.items():
            rgba = cam.get_rgba()
            if rgba is None:
                continue
            out[name] = np.asarray(rgba)[..., :3].astype(np.uint8)
        return out

    def get_obs(self) -> dict:
        obs = {
            "state":   self.get_state(),
            "joint_q": {s: self.get_arm_q(s) for s in self.names},
            "gripper": {s: self.get_gripper(s) for s in self.names},
        }
        if self.use_cameras:
            obs["images"] = self.get_images()
        return obs

    # ------------------------------------------------------------------
    # kinematics helpers (world frame)
    # ------------------------------------------------------------------
    def ik(self, side: str, pos_world, rot_world=None, q0=None, **kw):
        k  = self.kin[side]
        R0 = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), k.base_yaw)
        p_local = R0.T @ (np.asarray(pos_world, dtype=float) - k.base_pos)
        R_local  = None if rot_world is None else R0.T @ rot_world
        return k.solve(p_local, R_local, q0=q0, **kw)

    def fk(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """TCP (pos, rot) in world frame computed from current joint angles."""
        k  = self.kin[side]
        p_local, R_local = k.fk(self.get_arm_q(side))
        R0 = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), k.base_yaw)
        return k.base_pos + R0 @ p_local, R0 @ R_local

    # ------------------------------------------------------------------
    # cloth helpers
    # ------------------------------------------------------------------
    def get_cloth_points(self) -> np.ndarray:
        """Return (N, 3) sim-mesh vertex positions (cloth_readback=True required)."""
        return self.cloth.get_points()

    def _capture_cloth_reference(self):
        """Record the flat cloth's near-half vertices as the fold baseline."""
        try:
            pts = self.get_cloth_points()
        except Exception:
            return
        if pts is None or len(pts) < 4:
            return
        y = np.asarray(pts[:, 1], dtype=float)
        # Reject a stale/invalid readback (e.g. all-zero points) so the fold
        # baseline is never captured from garbage.
        span_x = float(pts[:, 0].max() - pts[:, 0].min())
        span_y = float(y.max() - y.min())
        z = np.asarray(pts[:, 2], dtype=float)
        half = float(self.cfg.cloth.size[0])
        if span_x < 0.5 * half or span_y < 0.5 * half:
            print(f"[RebotClothEnv] cloth reference skipped: invalid readback "
                  f"(span_x={span_x:.3f}, span_y={span_y:.3f})")
            return
        if z.min() < self.cfg.table.height - 0.15 or z.max() > self.cfg.table.height + 0.35:
            print(f"[RebotClothEnv] cloth reference skipped: z out of range "
                  f"[{z.min():.3f},{z.max():.3f}]")
            return
        yc = 0.5 * (float(y.min()) + float(y.max()))
        mask = y < yc
        if not np.any(mask):
            return
        self._cloth_ref_y = y.copy()
        self._cloth_near_mask = mask
        # vertices that start on the near edge (within 5 cm of y_min)
        self._cloth_edge_mask = y < (float(y.min()) + 0.05)
        # vertices that start on the far edge (within 5 cm of y_max)
        self._cloth_far_mask = y > (float(y.max()) - 0.05)
        self._cloth_ref_center = yc
        # after a half-fold the near-half centroid lands at +L/4
        self._cloth_ref_half = 0.25 * float(y.max() - y.min())
        self._cloth_near_start = float(y[mask].mean())
        self._cloth_near_target = self._cloth_ref_center + self._cloth_ref_half

    def measure_fold(self) -> float:
        """How far the near half has been folded over, in [0, 1].

        0 = flat, 1 = near half centred over the far half.  Tracked by the
        mean y of the vertices that start on the near half, which is robust to
        the sheet being lifted, carried and dropped.
        """
        if not getattr(self, "cloth_readback", False):
            return 0.0
        if self._cloth_ref_y is None:
            self._capture_cloth_reference()
            return 0.0
        try:
            pts = self.get_cloth_points()
        except Exception:
            return 0.0
        if pts is None or len(pts) != len(self._cloth_near_mask):
            return 0.0
        near = float(np.asarray(pts[:, 1])[self._cloth_near_mask].mean())
        denom = self._cloth_near_target - self._cloth_near_start
        return float(np.clip((near - self._cloth_near_start) / max(1e-6, denom), 0.0, 1.0))

    def measure_edge_alignment(self) -> float:
        """Gap (m) between the folded near edge and the far edge.

        0 means the near edge landed exactly on the far edge.  Positive means
        it stopped short; negative means it overshot past the far edge.
        """
        if not getattr(self, "cloth_readback", False):
            return 999.0
        if getattr(self, "_cloth_edge_mask", None) is None:
            return 999.0
        try:
            pts = self.get_cloth_points()
        except Exception:
            return 999.0
        if pts is None or len(pts) != len(self._cloth_edge_mask):
            return 999.0
        s = self.measure_edge_stats()
        return s.get("gap_mean", 999.0)

    def measure_edge_stats(self) -> dict:
        """Per-column gap between the folded near edge and the far edge.

        For each strip across the cloth width, compare the outermost far-half
        vertex with the outermost folded near-half vertex.  This is robust to
        wrinkles and to the whole cloth sliding, and exposes any yaw (the gap
        varying across x).
        """
        if not getattr(self, "cloth_readback", False):
            return {}
        if getattr(self, "_cloth_edge_mask", None) is None:
            return {}
        if getattr(self, "_cloth_far_mask", None) is None:
            return {}
        try:
            pts = self.get_cloth_points()
        except Exception:
            return {}
        if pts is None or len(pts) != len(self._cloth_edge_mask):
            return {}
        x = np.asarray(pts[:, 0])
        y = np.asarray(pts[:, 1])
        near = self._cloth_edge_mask
        far = self._cloth_far_mask
        xmin, xmax = float(x.min()), float(x.max())
        nbin = 8
        edges = np.linspace(xmin, xmax, nbin + 1)
        prof, gaps = [], []
        for i in range(nbin):
            m = (x >= edges[i]) & (x < edges[i + 1])
            mn = m & near
            mf = m & far
            if mn.sum() < 1 or mf.sum() < 1:
                continue
            near_y = float(y[mn].max())
            far_y = float(y[mf].max())
            gap = far_y - near_y
            gaps.append(gap)
            prof.append((round(float(0.5 * (edges[i] + edges[i + 1])), 3),
                         round(near_y, 3), round(far_y, 3)))
        if not gaps:
            return {}
        gaps = np.asarray(gaps)
        return {
            "gap_mean": float(gaps.mean()),
            "gap_mean_cm": float(abs(gaps.mean()) * 100.0),
            "gap_std_cm": float(gaps.std() * 100.0),
            "gap_max_cm": float(np.abs(gaps).max() * 100.0),
            "profile": prof,
        }

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, render: bool | None = None, randomise: bool = False):
        """Reset to home pose; optionally apply cloth randomisation.

        Note: we intentionally do NOT call world.reset() here — the reBot USD
        has PhysicsArticulationRootAPI already applied during URDF→USD conversion,
        and world.reset() re-initializes the physics scene which triggers
        "Nested articulation roots are not allowed" errors.  Instead we set
        joint positions directly and step the world.
        """
        render = self.use_cameras if render is None else render
        for side in self.names:
            q0 = self.kin[side].home_q
            q  = np.zeros(len(self.dof_names), dtype=np.float32)
            q[self.arm_idx]    = q0
            q[self.finger_idx] = self.rcfg.finger_open   # start open
            self.robots[side].set_joint_positions(self._arr(q))
        if randomise:
            self._randomise_cloth()
        else:
            # Re-roll the cloth to its initial flat state so physics state from
            # the previous episode does not carry over.  NB: the cloth mesh
            # points are authored in world coordinates and the root xform is
            # identity, so we must NOT translate the root prim here.
            try:
                self.cloth.restore()
            except Exception as exc:
                print(f"[RebotClothEnv] cloth reset note: {exc}")
        self.world.step(render=render)
        # capture the flat-cloth fold baseline after the reset step
        self._cloth_ref_y = None
        self._capture_cloth_reference()

    def _randomise_cloth(self):
        """Randomise cloth friction only (position/orientation fixed for policy stability)."""
        c   = self.cfg.cloth
        rng = self.rng
        # friction jitter (±20% around nominal)
        dfric = float(rng.uniform(0.8, 1.2))

        try:
            from pxr import Gf, UsdGeom
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            root  = stage.GetPrimAtPath(self.cloth.root_path)
            from .scene import _set_xform
            # keep cloth at default position/orientation; only change friction
            mat_prim = stage.GetPrimAtPath("/World/Looks/ClothDeformable")
            if mat_prim and mat_prim.HasAttribute("physxDeformableMaterial:dynamicFriction"):
                mat_prim.GetAttribute("physxDeformableMaterial:dynamicFriction").Set(
                    c.dynamic_friction * dfric
                )
        except Exception as exc:
            print(f"[RebotClothEnv] cloth randomise note: {exc}")

    # ------------------------------------------------------------------
    def close(self):
        try:
            self.app.close()
        except Exception:
            pass
