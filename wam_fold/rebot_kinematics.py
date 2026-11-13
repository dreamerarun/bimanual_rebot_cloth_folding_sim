"""reBot Arm T701 HT (B601) kinematics — built on the same SerialChain / ArmKinematics
framework as wam_fold.kinematics, but targeting the reBot B601 with 6 revolute arm
joints + 2 prismatic gripper fingers.

Joint map (from URDF):
  joint1 .. joint6   : revolute arm joints (the same order as reBot URDF)
  gripper_joint1/2   : prismatic fingers (stroke 0 → 0.05 m; closed=0, open≈0.045)

The ``ArmKinematics`` from wam_fold.kinematics works out‑of‑the-box with the
reBot URDF (it extracts 6 revolute joints).  This module adds task-specific
helpers for folding: TCP rotation conventions, grasp poses, and a simple
position‑based FOLD primitive.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field

import numpy as np

from wam_fold.kinematics import ArmKinematics, SerialChain, Joint, rpy_to_matrix, matrix_to_rpy, rot_error


def _to_np(x):
    """Works whether the backend returns numpy, torch, or warp arrays."""
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# reBot-specific URDF paths (set once at import time)
# ---------------------------------------------------------------------------
# Override the default URDF / TCP link after the chain has been built.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_rebot_urdf = str(_PROJECT_ROOT / "assets" / "rebot_t701" / "urdf" / "RS" / "urdf" / "ReBot_Arm_RS.urdf")
_rebot_tcp_link = "gripper_end"

if not pathlib.Path(_rebot_urdf).exists():
    raise FileNotFoundError(
        f"reBot URDF not found at {_rebot_urdf}. "
        "Run `python scripts/fetch_assets.py` from the project root first."
    )

# Re-build the chain with the reBot-specific TCP link
_rebot_chain = ArmKinematics(_rebot_urdf, tcp_link=_rebot_tcp_link)

# Expose the most useful bits at module level for convenience
num_dof = _rebot_chain.num_dof
dof_names = _rebot_chain.dof_names  # ['joint1', 'joint2', ..., 'gripper_joint1', ...]
lower = _to_np(_rebot_chain.lower)  # 6 arm joints
upper = _to_np(_rebot_chain.upper)  # 6 arm joints


# ---------------------------------------------------------------------------
# reBot home / neutral config
# ---------------------------------------------------------------------------
# A neutral configuration that keeps the arm above the table with elbow up.
# The six joints are tuned so the forearm is roughly horizontal and the
# TCP points down toward the table centre.
def _default_home() -> np.ndarray:
    q = np.zeros(_rebot_chain.num_dof)
    # The reBot URDF joints are named joint1..joint6 in order; assign neutral-ish angles.
    # These values were found by simple trial keeping the hand above the table centre.
    q[0] = 0.0          # joint1: base yaw
    q[1] = -1.2         # joint2: shoulder pitch (elbow up)
    q[2] = 0.0          # joint3: elbow pitch
    q[3] = -1.57        # joint4: wrist pitch (point down)
    q[4] = 0.0          # joint5: wrist roll
    q[5] = 1.57         # joint6: wrist pitch (slight bend)
    return np.clip(q, _rebot_chain.lower + 1e-3, _rebot_chain.upper - 1e-3)


home_q = _default_home()

# ---------------------------------------------------------------------------
# TCP rotation helpers
# ---------------------------------------------------------------------------
def tcp_rotation_facing(yaw: float = 0.0, tilt: float = 0.0) -> np.ndarray:
    """Rotation matrix for a grasp with tool Z pointing roughly at the table.

    Conventions (same as wam_fold.env.tcp_rotation_facing):
      - tool +Z should point toward the table (world -Z)
      - tool +X stays aligned with world X
      - yaw rotates around world Y; tilt pitches the tool about its +X axis
    """
    Rz = np.array(
        [[math.cos(yaw), 0, math.sin(yaw)],
         [0, 1, 0],
         [-math.sin(yaw), 0, math.cos(yaw)]],
        dtype=float,
    )
    Rx = np.array(
        [[1, 0, 0],
         [0, math.cos(tilt), -math.sin(tilt)],
         [0, math.sin(tilt), math.cos(tilt)]],
        dtype=float,
    )
    return Rz @ Rx


# ---------------------------------------------------------------------------
# Grasp / fold primitives
# ---------------------------------------------------------------------------
@dataclass
class GraspPose:
    """A grasp defined by TCP position (metres) and rotation (3×3 matrix)."""
    pos: np.ndarray  # shape (3,), world frame
    rot: np.ndarray  # shape (3, 3), world frame


# reBot grasp poses relative to table centre (0,0,0 on table top surface)
# These are initialisations for the "press-and-drag" fold.
# Units: metres for position, rotation matrix for rot.
# The left arm approaches from the table left, the right from the right.

# Press configuration (near edge of cloth):
left_press_approach = GraspPose(
    pos=np.array([-0.24, 0.24, 0.775], dtype=float),   # above near edge, left side
    rot=tcp_rotation_facing(yaw=0.0, tilt=0.0),
)

right_press_approach = GraspPose(
    pos=np.array([0.24, 0.24, 0.775], dtype=float),    # above near edge, right side
    rot=tcp_rotation_facing(yaw=3.14159265358979, tilt=0.0),
)

# Lift configuration (lift the grasped edge):
lift_height = 0.12  # metres above the press pose z

# Fold configuration (drag the far edge over the pressed edge):
# right arm drags the far edge over the left-pressed edge
fold_speed = 0.10  # metres per control cycle (slow, stable for 6-GB VRAM)
fold_fraction = 0.82  # fraction of far-edge travel relative to cloth width

# ---------------------------------------------------------------------------
# Bimanual fold task primitives (position-based; IK called externally)
# ---------------------------------------------------------------------------


def fold_primitives(left_target_pos: np.ndarray | None = None,
                    right_target_pos: np.ndarray | None = None,
                    lift: float = lift_height,
                    fold: float | None = None) -> dict:
    """Return IK solutions for the bimanual fold given optional overrides.

    If *left_target_pos* / *right_target_pos* are None, the default approach/
    fold poses defined above are used.  The function calls each arm's IK and
    returns joint position commands (7 values per arm = 6 arm + 1 gripper).

    Returns
    -------
    dict with keys ``left``, ``right`` each mapping to a tuple
    ``(joint_positions, gripper_open)`` where *joint_positions* is a length-6
    numpy array and *gripper_open* is a float in [0, 1] (0 = closed, 1 = open).
    """
    from wam_fold.env import look_at_quat

    # ---- default poses if not overridden ----
    if left_target_pos is None:
        left_target_pos = left_press_approach.pos.copy()
    if right_target_pos is None:
        right_target_pos = right_press_approach.pos.copy()

    # ---- build desired rotations: both arms face downward / toward table ----
    # The reBot's default tcp_rotation faces -Z (into the table); we keep it simple
    # and just pass the yaw from the GraspPose convention.
    left_rot = tcp_rotation_facing(yaw=0.0, tilt=0.0)
    right_rot = tcp_rotation_facing(yaw=np.pi, tilt=0.0)

    # ---- IK for each arm (using the shared ArmKinematics) ----
    k = _rebot_chain  # reBot ArmKinematics instance

    left_q, left_info = k.ik(
        target_pos=np.array(left_target_pos, dtype=float),
        target_rot=left_rot,
        q0=home_q,
        max_iters=200,
        tol_pos=1e-3,
        tol_rot=1e-2,
        pos_gain=1.0,
        rot_gain=0.6,
        damping=0.03,
        nullspace_gain=0.0,
    )

    right_q, right_info = k.ik(
        target_pos=np.array(right_target_pos, dtype=float),
        target_rot=right_rot,
        q0=home_q,
        max_iters=200,
        tol_pos=1e-3,
        tol_rot=1e-2,
        pos_gain=1.0,
        rot_gain=0.6,
        damping=0.03,
        nullspace_gain=0.0,
    )

    # ---- gripper command: closed = 0, open = 1 (we close on the cloth edge) ----
    gripper_open = 0.0  # start closed; will open on retreat

    return {
        "left": (left_q, gripper_open),
        "right": (right_q, gripper_open),
        "ik_info": {"left": left_info, "right": right_info},
    }


# ---------------------------------------------------------------------------
# Convenience: single-fold primitive (left presses, right drags over)
# ---------------------------------------------------------------------------
def single_fold_primitive() -> dict:
    """Convenience: left presses near edge, right drags far edge over it.

    Returns IK solutions ready to be sent as actions to the simulation.
    """
    return fold_primitives()


# ---------------------------------------------------------------------------
# Double-fold: both arms press and drag symmetrically
# ---------------------------------------------------------------------------
def double_fold_primitive() -> dict:
    """Convenience: both arms press near their respective edges and drag inward.

    Returns IK solutions for a symmetric double-fold.
    """
    # left presses near left edge, right presses near right edge
    left_pos = np.array([-0.28, 0.0, 0.775], dtype=float)
    right_pos = np.array([0.28, 0.0, 0.775], dtype=float)
    return fold_primitives(left_target_pos=left_pos, right_target_pos=right_pos)