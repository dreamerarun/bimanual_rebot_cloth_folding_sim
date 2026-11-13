"""ControlPolicy — smooth bimanual fold controller for the reBot cloth task.

Motion design
-------------
Per-step IK + exponential smoothing produced lag and chatter.  Instead this
policy *plans* every motion once in joint space:

* Each waypoint is reached with a single IK solve (seeded from the previous
  solution, so the joint-space path stays continuous).
* Straight moves interpolate between the start and target joint vectors with a
  quintic minimum-jerk profile (zero velocity and acceleration at both ends).
* The fold arc is a dense Cartesian path (carry over + lower) solved in one
  sequential IK pass, lightly smoothed, then played back over a fixed duration.
* The gripper open/close command is ramped smoothly over several control ticks.

This yields smooth, repeatable, accurate arm motion at 20 Hz control /
120 Hz physics, with no per-step solve jitter.

Action layout (14 dims):
    [q_arm_left(6), grip_left(1), q_arm_right(6), grip_right(1)]
Gripper convention: 0 = closed, 1 = open.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable

import numpy as np

from .config import FoldTaskCfg
from .kinematics import axis_angle_to_matrix


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class FoldState(IntEnum):
    IDLE     = 0
    APPROACH = 1
    GRASP    = 2
    LIFT     = 3
    FOLD     = 4
    PLACE    = 5
    RELEASE  = 6
    RETREAT  = 7
    DONE     = 8


STATE_NAMES = [s.name for s in FoldState]


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------
def min_jerk(t: float) -> float:
    """Quintic minimum-jerk time scaling on [0, 1]."""
    t = min(1.0, max(0.0, float(t)))
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def smooth_step(t: float) -> float:
    t = min(1.0, max(0.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _tool_orientation(base_yaw: float, approach_deg: float) -> np.ndarray:
    """World rotation for a top-down grasp tilted ``approach_deg`` from
    horizontal.  Tool X points along the approach direction, tool Y is the
    (horizontal) finger closing axis."""
    a = math.radians(approach_deg)
    x = np.array([math.cos(a), 0.0, -math.sin(a)])
    y = np.array([0.0, 1.0, 0.0])
    z = np.cross(x, y)
    R_local = np.column_stack([x, y, z])
    R0 = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), base_yaw)
    return R0 @ R_local


def _smooth_joint_path(path: np.ndarray, passes: int = 2) -> np.ndarray:
    """Light 3-tap moving average that preserves the endpoints."""
    out = np.asarray(path, dtype=float).copy()
    if len(out) < 3:
        return out
    for _ in range(max(0, passes)):
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = sm
    return out


# ---------------------------------------------------------------------------
# Per-step data record (stored for LeRobot dataset later)
# ---------------------------------------------------------------------------
@dataclass
class StepRecord:
    step:       int
    state_name: str
    action:     np.ndarray
    joint_q:    dict
    gripper:    dict
    ee_pose:    dict
    images:     dict | None = None
    fold_error: float     = 0.0
    success:    bool      = False


@dataclass
class EpisodeData:
    episode_id:    int
    steps:         list[StepRecord] = field(default_factory=list)
    success:       bool  = False
    fold_error_cm: float = 999.0
    duration_s:    float = 0.0
    config:        dict  = field(default_factory=dict)

    def __len__(self):
        return len(self.steps)


# ---------------------------------------------------------------------------
# ControlPolicy
# ---------------------------------------------------------------------------
class ControlPolicy:
    """Waypoint-scheduled, joint-space trajectory fold controller."""

    # nominal phase durations (control ticks @ 20 Hz)
    IDLE_STEPS      = 10
    APPROACH_STEPS  = 40
    DESCEND_STEPS   = 30
    CLOSE_STEPS     = 26
    GRASP_HOLD_STEPS = 12
    LIFT_STEPS      = 8
    FOLD_STEPS      = 140
    FOLD_HOLD_STEPS = 15
    RELEASE_STEPS   = 24
    RETREAT_STEPS   = 40

    def __init__(
        self,
        task_cfg:        FoldTaskCfg | None = None,
        control_hz:       float = 20.0,
        lift_height:     float | None = None,
        grasp_close_val: float = 0.0,
        grasp_open_val:  float = 1.0,
    ):
        self.task_cfg    = task_cfg or FoldTaskCfg()
        self.control_hz  = control_hz
        self.lift_height = lift_height if lift_height is not None \
            else self.task_cfg.lift_height
        self.grasp_close = grasp_close_val
        self.grasp_open  = grasp_open_val

        self.state: FoldState = FoldState.IDLE
        self.state_step: int  = 0
        self.total_step: int  = 0

        # planned trajectories, filled by prepare()
        self._plan: dict[str, dict] = {}
        self._grasp_geom: dict[str, Any] = {}

        # last commanded joint vector per side (used as segment start)
        self._q_hold: dict[str, np.ndarray | None] = {"left": None, "right": None}

        # active joint-space segment
        self._seg: dict | None = None
        self._seg_step: int = 0
        self._phase: int = 0

        # active dense path (fold)
        self._path: np.ndarray | None = None
        self._path_side: str | None = None

        self.on_state_change: Callable[[FoldState, FoldState], None] | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def reset(self, env):
        self.state      = FoldState.IDLE
        self.state_step = 0
        self.total_step = 0
        self._seg       = None
        self._seg_step  = 0
        self._path      = None
        self.prepare(env)
        for side in env.names:
            q0 = env.get_arm_q(side).copy()
            self._q_hold[side] = q0

    def prepare(self, env):
        """Precompute every waypoint trajectory for the current cloth pose."""
        names = tuple(env.names)
        geom = self._cloth_geometry(env)
        self._grasp_geom = geom

        plans: dict[str, dict] = {}
        for side in names:
            plans[side] = self._plan_side(env, side, geom)
        self._plan = plans

        # report any waypoint that IK could not reach
        for side in names:
            for key in ("hover", "grasp", "lift", "retreat"):
                err = plans[side][f"{key}_err"]
                if err > 0.01:
                    print(f"[policy] WARNING {side} {key} IK error {err*1000:.1f} mm")

    def get_action(self, env) -> np.ndarray:
        action = np.zeros(env.ACTION_DIM, dtype=np.float32)
        handler = self._state_handlers.get(self.state)
        if handler is None:
            self._hold(env, action)
        else:
            handler(self, env, action)
        self.state_step += 1
        self.total_step  += 1
        return action

    @property
    def is_done(self) -> bool:
        return self.state == FoldState.DONE

    # ------------------------------------------------------------------
    # geometry / planning
    # ------------------------------------------------------------------
    def _cloth_geometry(self, env) -> dict:
        tc = self.task_cfg
        xmin, xmax = -tc.cloth_half, tc.cloth_half
        ymin, ymax = -tc.cloth_half, tc.cloth_half
        surf_z = tc.table_height + 0.006
        pts = None
        if getattr(env, "cloth_readback", False):
            try:
                pts = env.get_cloth_points()
            except Exception:
                pts = None
        if pts is not None and len(pts) > 4:
            # Guard against a stale/empty readback (all-zero points would put
            # the grasp targets at the world origin and blow up the IK).
            span_x = float(pts[:, 0].max() - pts[:, 0].min())
            span_y = float(pts[:, 1].max() - pts[:, 1].min())
            z_lo   = tc.table_height - 0.15
            z_hi   = tc.table_height + 0.35
            z_ok   = float(pts[:, 2].min()) > z_lo and float(pts[:, 2].max()) < z_hi
            if span_x < 0.5 * 2 * tc.cloth_half or span_y < 0.5 * 2 * tc.cloth_half or not z_ok:
                print(f"[policy] WARNING cloth readback looks invalid "
                      f"(span_x={span_x:.3f}, span_y={span_y:.3f}, "
                      f"z=[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]); "
                      f"falling back to nominal cloth geometry")
                pts = None
        if pts is not None and len(pts) > 4:
            xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
            ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
            # 75th percentile ≈ the flat, supported sheet (ignores the drooping
            # overhang if the cloth sits on a support board)
            surf_z = float(np.percentile(pts[:, 2], 75))
        width = max(1e-3, xmax - xmin)
        inset = tc.grasp_side_inset_frac * width
        left_x = xmin + inset
        right_x = xmax - inset
        grasp_y = ymin + tc.grasp_edge_inset
        # Fold line is the cloth midline.  Mirroring the grasp point across it
        # puts the pinched near edge exactly on the far edge.
        midline_y = 0.5 * (ymin + ymax)
        fold_y = 2.0 * midline_y - grasp_y

        # The sheet lies flat on the table; the pad bottoms sit at the TCP, so
        # the grasp height is just above the sheet's mid-surface.
        edge_z = surf_z
        if pts is not None and len(pts) > 8:
            sel = (np.abs(pts[:, 0] - left_x) < 0.06) & (pts[:, 1] < ymin + 0.05)
            if sel.sum() < 3:
                sel = pts[:, 1] < ymin + 0.05
            if sel.sum() >= 3:
                edge_z = float(np.percentile(pts[sel, 2], 50))
        grasp_z = edge_z + tc.grasp_clearance
        place_z = surf_z + tc.place_clearance

        return {
            "left_x":  left_x,
            "right_x": right_x,
            "grasp_y": grasp_y,
            "fold_y":  fold_y,
            "midline_y": midline_y,
            "far_edge_y": ymax,
            "surface_z": surf_z,
            "edge_z": edge_z,
            "support": False,
            "grasp_z": grasp_z,
            "place_z": place_z,
            "finger_z_corr": tc.finger_z_corr,
        }

    def _plan_side(self, env, side: str, geom: dict) -> dict:
        """Plan hover -> pinch -> lift -> carry over -> lower -> retreat.

        This is a true grasp-and-fold: the fingers close on the near edge,
        lift the near half off the table, carry it over the far half and set
        it down.  The two arms hold the edge between them for the whole carry.

        The right finger collision mesh sits ~1.5 mm higher than the left
        relative to the TCP, so the right grasp is lowered by that much to
        straddle the same edge.
        """
        tc = self.task_cfg
        kin = env.kin[side]
        R = _tool_orientation(kin.base_yaw, tc.approach_deg)
        gx = geom["left_x"] if side == "left" else geom["right_x"]
        py = geom["grasp_y"]
        pz = geom["grasp_z"] + (0.0 if side == "left" else geom["finger_z_corr"])
        fy = geom["fold_y"]
        lift_z = pz + self.lift_height
        place_z = geom["place_z"]

        def ik(pos, seed):
            q, info = env.ik(side, np.asarray(pos, dtype=float), R, q0=seed,
                             max_iters=800, tol_pos=5e-4, tol_rot=1e-2,
                             nullspace_gain=0.02, step_limit=0.2)
            return np.asarray(q, dtype=float), float(info.get("pos_err", 999.0))

        # Seed from the *actual* settled joint state: the all-zero home_q is a
        # singular configuration and the solver can stall in a local minimum.
        q_home = np.asarray(env.get_arm_q(side), dtype=float).copy()
        q_hover, e_hover = ik((gx, py, pz + tc.hover_height), q_home)
        q_grasp, e_grasp = ik((gx, py, pz), q_hover)
        q_lift, e_lift = ik((gx, py, lift_z), q_grasp)

        # Carry the pinched edge along a circular arc about the fold line
        # (the cloth midline).  The near half is r long, so the pinch must
        # stay exactly r from the fold line; a constant-height sweep pulls the
        # far half forward and the fold lands short.  The arc lifts the pinch
        # over the top and sets it down on the far edge with no over-stretch.
        midline_y = geom["midline_y"]
        zc = geom["surface_z"]
        r0 = float(np.hypot(py - midline_y, pz - zc))  # pinch radius from fold line
        # A rigid half-fold keeps the pinch at a constant radius from the fold
        # line.  Carrying at constant height instead over-stretches the near
        # half and drags the far half forward.  The end radius is shortened a
        # little because the whole sheet drifts under the fold; the near edge
        # then lands on the far edge instead of overrunning it.
        r1 = max(0.02, r0 - tc.fold_overshoot)
        n_path = 49
        q_carry = []
        last_e = 0.0
        for i in range(n_path):
            theta = math.pi * i / (n_path - 1)
            rr = r0 + (r1 - r0) * smooth_step(theta / math.pi)
            y = midline_y - rr * math.cos(theta)
            z = zc + rr * math.sin(theta) + (place_z - zc) * (theta / math.pi) ** 2
            seed = q_lift if i == 0 else q_carry[-1]
            q, e = ik((gx, y, z), seed)
            q_carry.append(q)
            last_e = max(last_e, e)
        q_carry_arr = _smooth_joint_path(np.asarray(q_carry), passes=2)
        # resample to FOLD_STEPS ticks with a min-jerk time parameter
        src = np.linspace(0.0, 1.0, len(q_carry_arr))
        dst = np.array([min_jerk(i / self.FOLD_STEPS) for i in range(self.FOLD_STEPS + 1)])
        carry_seq = np.empty((len(dst), q_carry_arr.shape[1]), dtype=float)
        for j in range(q_carry_arr.shape[1]):
            carry_seq[:, j] = np.interp(dst, src, q_carry_arr[:, j])

        # lift clear with the fingers closed, then open in the air
        q_release, e_rel = ik((gx, fy, place_z + 0.030), q_carry[-1])
        # then rise straight up off the fold
        q_retreat, e_ret = ik((gx, fy, place_z + 0.22), q_carry[-1])

        return {
            "hover": q_hover, "hover_err": e_hover,
            "grasp": q_grasp, "grasp_err": e_grasp,
            "lift": q_lift, "lift_err": e_lift,
            "carry": carry_seq, "carry_err": last_e,
            "release": q_release, "release_err": e_rel,
            "retreat": q_retreat, "retreat_err": e_ret,
            "R": R,
        }

    # ------------------------------------------------------------------
    # action helpers
    # ------------------------------------------------------------------
    def _transition(self, new_state: FoldState):
        old = self.state
        if old != new_state and self.on_state_change:
            self.on_state_change(old, new_state)
        self.state      = new_state
        self.state_step = 0
        self._seg       = None
        self._seg_step  = 0
        self._phase     = 0

    def _write(self, action, env, q: dict, grip: dict):
        for k, side in enumerate(env.names):
            self._q_hold[side] = np.asarray(q[side], dtype=float)
            action[k * 7:k * 7 + 6] = q[side]
            action[k * 7 + 6]      = float(np.clip(grip[side], 0.0, 1.0))

    def _hold(self, env, action):
        q = {s: (self._q_hold[s] if self._q_hold[s] is not None else env.get_arm_q(s))
             for s in env.names}
        self._write(action, env, q, {s: self.grasp_open for s in env.names})

    def _start_segment(self, env, targets: dict, steps: int,
                       grip0: float, grip1: float):
        self._seg = {
            "q0": {s: np.asarray(self._q_hold[s], dtype=float) for s in env.names},
            "q1": {s: np.asarray(targets[s], dtype=float) for s in env.names},
            "n": max(1, int(steps)),
            "g0": float(grip0),
            "g1": float(grip1),
        }
        self._seg_step = 0

    def _play_segment(self, env, action) -> bool:
        """Advance the active min-jerk segment.  Returns True when finished."""
        seg = self._seg
        n = seg["n"]
        s = min_jerk(self._seg_step / n)
        q = {side: seg["q0"][side] + (seg["q1"][side] - seg["q0"][side]) * s
             for side in env.names}
        g = seg["g0"] + (seg["g1"] - seg["g0"]) * smooth_step(s)
        self._write(action, env, q, {side: g for side in env.names})
        self._seg_step += 1
        if self._seg_step > n:
            self._seg = None
            return True
        return False

    # ------------------------------------------------------------------
    # states
    # ------------------------------------------------------------------
    def _h_idle(self, env, action):
        self._hold(env, action)
        if self.state_step >= self.IDLE_STEPS:
            self._transition(FoldState.APPROACH)

    def _h_approach(self, env, action):
        # fly to the hover pose while closing the gripper to make a paddle
        if self._seg is None:
            self._start_segment(env, {s: self._plan[s]["hover"] for s in env.names},
                                self.APPROACH_STEPS, self.grasp_open, self.grasp_open)
        done = self._play_segment(env, action)
        if done and self.state_step >= self.APPROACH_STEPS:
            self._transition(FoldState.GRASP)

    def _h_grasp(self, env, action):
        targets = {s: self._plan[s]["grasp"] for s in env.names}
        if self._phase == 0:
            # descend with the fingers open so the pads straddle the edge
            self._start_segment(env, targets, self.DESCEND_STEPS,
                                self.grasp_open, self.grasp_open)
            self._phase = 1
            self._play_segment(env, action)
            return
        if self._phase == 1:
            if self._play_segment(env, action):
                self._phase = 2
            return
        if self._phase == 2:
            # close in place, clamping the edge
            self._start_segment(env, targets, self.CLOSE_STEPS,
                                self.grasp_open, self.grasp_close)
            self._phase = 3
            self._play_segment(env, action)
            return
        if self._phase == 3:
            if self._play_segment(env, action):
                self._phase = 4
                self._seg_step = 0
            return
        # phase 4: hold clamped so the pads settle on the edge, then lift
        self._hold(env, action)
        for k in range(len(env.names)):
            action[k * 7 + 6] = self.grasp_close
        self._seg_step += 1
        if self._seg_step >= self.GRASP_HOLD_STEPS:
            self._transition(FoldState.LIFT)

    def _h_lift(self, env, action):
        if self._seg is None:
            self._start_segment(env, {s: self._plan[s]["lift"] for s in env.names},
                                self.LIFT_STEPS, self.grasp_close, self.grasp_close)
        done = self._play_segment(env, action)
        if done and self.state_step >= self.LIFT_STEPS:
            self._transition(FoldState.FOLD)

    def _h_fold(self, env, action):
        if self._path is None:
            # both arms play their precomputed dense joint path in lock-step
            self._path = {s: self._plan[s]["carry"] for s in env.names}
            self._seg_step = 0
        idx = min(self._seg_step, self.FOLD_STEPS)
        q = {s: self._path[s][idx] for s in env.names}
        self._write(action, env, q, {s: self.grasp_close for s in env.names})
        self._seg_step += 1
        if self._seg_step > self.FOLD_STEPS + self.FOLD_HOLD_STEPS:
            self._path = None
            self._transition(FoldState.RELEASE)

    def _h_release(self, env, action):
        # lift clear with the fingers closed, then open in the air so the pads
        # never push the fold sideways
        targets = {s: self._plan[s]["release"] for s in env.names}
        if self._phase == 0:
            self._start_segment(env, targets, self.RELEASE_STEPS // 2,
                                self.grasp_close, self.grasp_close)
            self._phase = 1
            self._play_segment(env, action)
            return
        if self._phase == 1:
            if self._play_segment(env, action):
                self._phase = 2
                self._seg_step = 0
            return
        # open the fingers at height
        self._hold(env, action)
        for k in range(len(env.names)):
            action[k * 7 + 6] = self.grasp_open
        self._seg_step += 1
        if self._seg_step >= self.RELEASE_STEPS // 2:
            self._transition(FoldState.RETREAT)

    def _h_place(self, env, action):
        # retained for API compatibility; FOLD lowers directly into RELEASE
        self._transition(FoldState.RELEASE)

    def _h_retreat(self, env, action):
        if self._seg is None:
            self._start_segment(env, {s: self._plan[s]["retreat"] for s in env.names},
                                self.RETREAT_STEPS, self.grasp_open, self.grasp_open)
        done = self._play_segment(env, action)
        if done and self.state_step >= self.RETREAT_STEPS:
            self._transition(FoldState.DONE)

    def _h_done(self, env, action):
        self._hold(env, action)

    _state_handlers: dict = None


ControlPolicy._state_handlers = {
    FoldState.IDLE:     ControlPolicy._h_idle,
    FoldState.APPROACH: ControlPolicy._h_approach,
    FoldState.GRASP:    ControlPolicy._h_grasp,
    FoldState.LIFT:     ControlPolicy._h_lift,
    FoldState.FOLD:     ControlPolicy._h_fold,
    FoldState.PLACE:    ControlPolicy._h_place,
    FoldState.RELEASE:  ControlPolicy._h_release,
    FoldState.RETREAT:  ControlPolicy._h_retreat,
    FoldState.DONE:     ControlPolicy._h_done,
}


# ---------------------------------------------------------------------------
# Success metric
# ---------------------------------------------------------------------------
def compute_fold_error_cm(env, target_fraction: float = 1.0) -> float:
    """Distance (cm) between the achieved fold and a complete half-fold.

    ``env.measure_fold()`` returns 0 for a flat sheet and 1 for a perfect
    half-fold; the residual is expressed as cm of missing travel of the
    near half (half the cloth length at most).
    """
    if not getattr(env, "cloth_readback", False):
        return 999.0
    try:
        current = env.measure_fold()
    except Exception:
        return 999.0
    error = max(0.0, min(1.0, target_fraction - current))
    cloth_len = env.cfg.cloth.size[1]
    return float(error * 0.5 * cloth_len * 100.0)


def check_success(env, target_fraction: float = 1.0, threshold_cm: float = 2.0) -> bool:
    return compute_fold_error_cm(env, target_fraction) < threshold_cm


# ---------------------------------------------------------------------------
# Episode recorder
# ---------------------------------------------------------------------------
class EpisodeRecorder:
    def __init__(self, episode_id: int, save_images: bool = True):
        self.episode_id  = episode_id
        self.save_images = save_images
        self.data = EpisodeData(episode_id=episode_id)
        self._t0  = time.time()

    def record(self, step: int, env, action: np.ndarray, state_name: str,
               fold_error_cm: float = 999.0, success: bool = False):
        joint_q = {s: env.get_arm_q(s).copy() for s in env.names}
        gripper = {s: env.get_gripper(s) for s in env.names}
        ee_pose = {}
        for s in env.names:
            pos, _ = env.fk(s)
            ee_pose[s] = pos.copy()

        images = None
        if self.save_images and env.use_cameras:
            images = env.get_images()
            images = {k: v.copy() for k, v in images.items()} if images else None

        self.data.steps.append(StepRecord(
            step=step,
            state_name=state_name,
            action=action.copy(),
            joint_q=joint_q,
            gripper=gripper,
            ee_pose=ee_pose,
            images=images,
            fold_error=fold_error_cm,
            success=success,
        ))

    def finalize(self, success: bool, fold_error_cm: float, config: dict | None = None):
        self.data.success       = success
        self.data.fold_error_cm = fold_error_cm
        self.data.duration_s    = time.time() - self._t0
        if config is not None:
            self.data.config = config
        return self.data
