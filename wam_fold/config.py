"""Global configuration for the two-arm WAM cloth-folding environment."""

from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass, field

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "assets"
WAM_USD = ASSET_DIR / "wam" / "wam_bhand_flat.usd"
WAM_URDF = ASSET_DIR / "urdf" / "wam_bhand.urdf"
TABLE_USD = ASSET_DIR / "nvidia" / "packing_table.usd"


@dataclass
class RobotCfg:
    usd_path: str = str(WAM_USD)
    urdf_path: str = str(WAM_URDF)
    tcp_link: str = "wam/bhand_grasp_link"
    # base pose in world (metres / radians about Z)
    left_base_xyz: tuple = (-0.80, 0.0, 0.75)
    right_base_xyz: tuple = (0.80, 0.0, 0.75)
    left_base_yaw: float = 0.0
    right_base_yaw: float = float(np.pi)
    # joint drive gains for the 7 arm joints
    stiffness: float = 2.0e4
    damping: float = 2.0e3
    # BarrettHand: (open, closed) angle for each finger joint type
    finger_curl: tuple = (("prox", 0.0, 1.45), ("med", 0.0, 1.30), ("dist", 0.0, 0.55))


@dataclass
class TableCfg:
    height: float = 0.75
    size: tuple = (1.7, 1.0, 0.05)   # x, y, z (top surface at height)
    use_usd: bool = False
    color: tuple = (0.55, 0.42, 0.30)


@dataclass
class ClothCfg:
    size: tuple = (0.52, 0.52)       # metres, x/y — slightly smaller to stay clear of arm bases
    center: tuple = (0.0, 0.0)       # metres, x/y (z follows table)
    resolution: int = 20             # particles per side
    # Thicker felt-like sheet: a 4 mm sheet is too thin for the block finger
    # pads to clamp on a tabletop, so the pinch slips.  10 mm gives the pads a
    # real edge to squeeze.
    thickness: float = 0.010
    mass: float = 0.18
    dynamic_friction: float = 4.0   # high grip so the pinch does not slip
    stretch_stiffness: float = 4000.0  # stiff enough that the far half does not compress
    shear_stiffness: float = 600.0
    bend_stiffness: float = 8.0
    subdivision_simplification: bool = False
    # z offset above the support surface — must be > thickness/2 so cloth sits ON it
    height_offset: float = 0.010
    # Low pedestal under the cloth.  Pinching a thin sheet that lies flat on a
    # table is geometrically impossible for block fingers (they hit the table
    # before closing).  A pedestal lifts the sheet, leaving a gap under the
    # overhanging edge so the finger pads can close around it.
    # Low support board.  The cloth overhangs it by a few cm and the near edge
    # drapes vertically over the side — that hanging flap is what the gripper
    # clamps (a flat sheet on a table cannot be pinched between the block pads).
    support_height: float = 0.0          # flat on the table (wipe primitive)
    support_size: tuple = (0.36, 0.36)
    # Small XY translation of the sheet at build time (metres).  This is applied
    # to the *mesh points* before the deformable hierarchy is cooked (never to
    # the root xform, whose points are already in world coordinates).  Used only
    # for demonstration diversity, sampled from RebotClothEnv.rng.
    jitter_xy: float = 0.0


@dataclass
class CameraCfg:
    name: str
    position: tuple
    look_at: tuple
    resolution: tuple = (256, 256)
    focal_length: float = 18.0
    attach_to: str | None = None    # prim path suffix for wrist cameras


@dataclass
class SceneCfg:
    physics_dt: float = 1.0 / 120.0
    rendering_dt: float = 1.0 / 60.0
    control_hz: float = 20.0
    robot: RobotCfg = field(default_factory=RobotCfg)
    table: TableCfg = field(default_factory=TableCfg)
    cloth: ClothCfg = field(default_factory=ClothCfg)
    cameras: tuple = (
        CameraCfg("top",   (0.0, 0.0, 1.55), (0.0, 0.0, 0.75), (256, 256), 10.0),
        CameraCfg("front", (1.35, -1.15, 1.25), (0.0, 0.0, 0.80), (256, 256), 22.0),
    )
    headless: bool = True
    rendering: bool = True


@dataclass
class FoldTaskCfg:
    """Geometry + timing for the two-arm Y-axis folding primitive (world).

    Motion: both arms grasp the NEAR y-edge of the cloth and carry it up and
    OVER to the FAR y-edge (a half-fold in Y).  Grasp points, surface height
    and fold target are derived from the live cloth mesh when readback is on,
    so the primitive survives small cloth placement changes.

    Validated setup: cloth 0.52×0.52 m, surface z≈0.752 m, arm bases at
    x=±0.55 m, tool tilted 75° from horizontal (closing axis along Y).
    """
    # Fallback geometry when cloth readback is unavailable
    cloth_half:     float = 0.26     # half-size of the square cloth (m)
    table_height:   float = 0.75     # table-top z (m)

    # How the grasp points are derived from the cloth mesh
    grasp_side_inset_frac: float = 0.18  # fraction of width in from each side
    grasp_edge_inset:      float = 0.013  # in from the near edge (pinch)
    fold_overshoot:        float = 0.030  # positive shortens the carry past the mirror
    flap_grasp_z:          float = 0.010  # legacy
    place_clearance:       float = 0.010  # TCP above the surface on drop (folded layer height)
    fold_edge_inset:       float = 0.060  # in from the far edge for the drop (m)
    # The pad bottoms sit at the TCP; grasp just above the sheet mid-surface so
    # the pads straddle the edge without jamming into the tabletop.
    grasp_clearance:       float = 0.001  # TCP above the cloth mid-surface (m)
    # right finger collision mesh is ~1.5 mm higher than the left wrt the TCP
    finger_z_corr:         float = -0.0015
    wipe_arc:              float = 0.006  # legacy
    hover_height:          float = 0.12   # pregrasp hover above the grasp (m)
    approach_deg:          float = 75.0   # tool tilt from horizontal (deg)
    retreat_dy:            float = -0.10  # retreat offset in +y before lifting

    # Motion
    lift_height:   float = 0.0       # arc carry lifts the pinch itself
    settle_steps:  int   = 30

    # Legacy fields (kept so older scripts still import cleanly)
    left_grasp:    tuple = (-0.20, -0.25, 0.760)
    right_grasp:   tuple = ( 0.20, -0.25, 0.760)
    fold_speed:    float = 0.10
    fold_y_target: float = 0.20
    fold_fraction: float = 0.80


@dataclass
class CollectCfg:
    out_dir: str = str(PROJECT_ROOT / "data" / "raw")
    episodes: int = 30
    seed: int = 0
    max_steps: int = 260
    image_size: int = 256
    save_video: bool = True
    camera_names: tuple = ("top", "front")


def as_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)
