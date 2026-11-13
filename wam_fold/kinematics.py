"""Forward kinematics + damped-least-squares IK for the WAM arm.

Everything here is pure numpy and is derived at import time from the URDF that
`scripts/01_build_robot_usd.py` produces, so the same code works for any arm
you swap in later (e.g. a 6-DOF YAM) as long as the URDF is available.

Why not use Isaac's Jacobian?  Sim-side Jacobians are only available after an
articulation is initialised and are expressed in the root frame.  Doing the
maths on the CPU lets us tele-operate at 30-60 Hz with several IK iterations
per tick without touching the physics engine, and makes the whole thing
testable without launching Isaac Sim.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# small SE(3) helpers
# ---------------------------------------------------------------------------
def rpy_to_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    p = math.asin(sy)
    if abs(sy) < 0.9999:
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        r = math.atan2(-R[1, 2], R[1, 1])
        y = 0.0
    return np.array([r, p, y])


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def rot_error(R_target: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Rotation error as an axis-angle vector in world frame."""
    R = R_target @ R_cur.T
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(tr)
    if angle < 1e-8:
        return np.zeros(3)
    if abs(angle - math.pi) < 1e-4:
        # 180 deg: use the largest diagonal element to recover the axis
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        axis[np.argmax(axis)] = axis.max()
        axis = axis / (np.linalg.norm(axis) + 1e-12)
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis / (2.0 * math.sin(angle))
    return axis * angle


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------
@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray | None
    lower: float
    upper: float


class SerialChain:
    """A serial chain extracted from a URDF, from a root link to a TCP link."""

    def __init__(self, urdf_path: str, tcp_link: str, root_link: str | None = None):
        root = ET.parse(urdf_path).getroot()
        joints = []
        for j in root.findall("joint"):
            origin = j.find("origin")
            axis = j.find("axis")
            lim = j.find("limit")
            joints.append(
                Joint(
                    name=j.get("name"),
                    kind=j.get("type"),
                    parent=j.find("parent").get("link"),
                    child=j.find("child").get("link"),
                    xyz=np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]),
                    rpy=np.array([float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]),
                    axis=None
                    if axis is None
                    else np.array([float(v) for v in axis.get("xyz", "0 0 1").split()]),
                    lower=float(lim.get("lower", -math.pi)) if lim is not None else -math.pi,
                    upper=float(lim.get("upper", math.pi)) if lim is not None else math.pi,
                )
            )
        by_child = {j.child: j for j in joints}

        chain = []
        link = tcp_link
        while link in by_child:
            j = by_child[link]
            chain.append(j)
            link = j.parent
        chain.reverse()
        if root_link is not None and chain and chain[0].parent != root_link:
            raise ValueError(f"chain does not start at {root_link} (starts at {chain[0].parent})")
        self.chain = chain
        self.urdf_path = urdf_path
        self.root_link = chain[0].parent if chain else None
        self.tcp_link = tcp_link

        self.revolute = [j for j in chain if j.kind in ("revolute", "continuous")]
        self.num_dof = len(self.revolute)
        self.dof_names = [j.name for j in self.revolute]
        self.lower = np.array([j.lower for j in self.revolute])
        self.upper = np.array([j.upper for j in self.revolute])

        # map dof -> index inside self.chain
        self._dof_chain_idx = [i for i, j in enumerate(chain) if j.kind in ("revolute", "continuous")]

    # -- forward kinematics -------------------------------------------------
    def fk(self, q: np.ndarray, link: str | None = None, return_frames: bool = False):
        """Return (position, rotation) of `link` (default: TCP).

        With return_frames=True also returns the per-DOF world axis/origin and
        the world transform of every link, which the Jacobian needs.
        """
        link = link or self.tcp_link
        T = np.eye(4)
        qi = 0
        axes, origins, frames = [], [], []
        for j in self.chain:
            T = T @ _transform(j.xyz, j.rpy)
            if j.kind in ("revolute", "continuous"):
                axes.append(T[:3, :3] @ j.axis)
                origins.append(T[:3, 3].copy())
                T = T @ _rot_axis(j.axis, q[qi])
                qi += 1
            frames.append((j.child, T.copy()))
            if j.child == link:
                break
        else:
            if link != self.chain[-1].child:
                raise KeyError(link)
        pos, rot = T[:3, 3].copy(), T[:3, :3].copy()
        if return_frames:
            return pos, rot, np.array(axes), np.array(origins), frames
        return pos, rot

    # -- analytic jacobian --------------------------------------------------
    def jacobian(self, q: np.ndarray, link: str | None = None) -> np.ndarray:
        p, _, axes, origins, _ = self.fk(q, link, return_frames=True)
        J = np.zeros((6, self.num_dof))
        for i in range(self.num_dof):
            J[:3, i] = np.cross(axes[i], p - origins[i])
            J[3:, i] = axes[i]
        return J

    # -- IK -----------------------------------------------------------------
    def ik(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray | None = None,
        q0: np.ndarray | None = None,
        home: np.ndarray | None = None,
        max_iters: int = 60,
        tol_pos: float = 5e-4,
        tol_rot: float = 5e-3,
        pos_gain: float = 1.0,
        rot_gain: float = 0.6,
        damping: float = 0.03,
        nullspace_gain: float = 0.15,
        step_limit: float = 0.25,
    ) -> tuple[np.ndarray, dict]:
        q = np.zeros(self.num_dof) if q0 is None else np.array(q0, dtype=float).copy()
        q = np.clip(q, self.lower + 1e-4, self.upper - 1e-4)
        home = q if home is None else home
        err_p = err_r = 9.9
        for it in range(max_iters):
            p, R = self.fk(q)
            dp = target_pos - p
            err_p = np.linalg.norm(dp)
            J = self.jacobian(q)
            if target_rot is None:
                J, e = J[:3], dp * pos_gain
                err_r = 0.0
            else:
                dr = rot_error(target_rot, R)
                err_r = np.linalg.norm(dr)
                J = np.vstack([J[:3], J[3:]])
                e = np.concatenate([dp * pos_gain, dr * rot_gain])
            if err_p < tol_pos and err_r < tol_rot:
                break
            # damped least squares
            JJt = J @ J.T + (damping**2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, e)
            # joint-limit aware nullspace posture bias
            if nullspace_gain > 0:
                N = np.eye(self.num_dof) - J.T @ np.linalg.solve(JJt, J)
                dq = dq + nullspace_gain * (N @ (home - q))
            m = np.max(np.abs(dq))
            if m > step_limit:
                dq *= step_limit / m
            q = np.clip(q + dq, self.lower + 1e-4, self.upper - 1e-4)
        p, R = self.fk(q)
        info = {
            "pos_err": float(np.linalg.norm(target_pos - p)),
        }
        if target_rot is not None:
            info["rot_err"] = float(np.linalg.norm(rot_error(target_rot, R)))
        info["iters"] = it + 1
        return q, info

    def fk_numpy_batch(self, qs: np.ndarray) -> np.ndarray:
        return np.stack([self.fk(q)[0] for q in qs])


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = axis_angle_to_matrix(axis, angle)
    return T


# ---------------------------------------------------------------------------
# convenience wrapper used by the env
# ---------------------------------------------------------------------------
class ArmKinematics(SerialChain):
    """WAM chain + handy helpers for table-top manipulation."""

    def __init__(self, urdf_path: str, tcp_link: str = "wam/bhand_grasp_link"):
        super().__init__(urdf_path, tcp_link)
        # A neutral, elbow-up configuration that keeps the arm above the table.
        # The WAM's zero pose points the forearm straight out; folding the
        # elbow a bit gives much better reachability over a table.
        self.home_q = self._default_home()

    def _default_home(self) -> np.ndarray:
        q = np.zeros(self.num_dof)
        # name based so it survives a swap of URDF
        for i, name in enumerate(self.dof_names):
            if "shoulder_pitch" in name:
                q[i] = 0.9
            elif "shoulder_yaw" in name:
                q[i] = 0.0
            elif "elbow_pitch" in name:
                q[i] = 1.6
            elif "wrist_pitch" in name:
                q[i] = 0.0
        return np.clip(q, self.lower + 1e-3, self.upper - 1e-3)

    def tcp_rotation_facing(self, yaw: float = 0.0, tilt: float = 0.0) -> np.ndarray:
        """Rotation matrix for a top-down grasp (tool Z pointing at the table)."""
        # palm/tool +Z should point world -Z; keep tool +X aligned with world X
        Rz = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), yaw)
        Rx = axis_angle_to_matrix(np.array([1.0, 0.0, 0.0]), math.pi + tilt)
        return Rz @ Rx

    def solve(self, target_pos, target_rot=None, q0=None, **kw):
        return self.ik(target_pos, target_rot, q0=q0, home=self.home_q, **kw)
