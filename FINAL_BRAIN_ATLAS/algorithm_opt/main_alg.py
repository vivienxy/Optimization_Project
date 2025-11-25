#!/usr/bin/env python3
"""
3D path planning for flexible needles – Algorithm 1 with full cost function.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from typing import Protocol
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import zoom
from vedo import Line, Points, show
from vedo import Volume

import numpy as np

# ---------------------------------------------------------------------------
# DEBUG / LOGGING
# ---------------------------------------------------------------------------

DEBUG = True  # master switch

PREVIEW_ENTRIES = False # sanity check for visualizing entry points
IGNORE_OBSTACLES = True  # simulate no obstacles

def log(msg: str) -> None:
    if DEBUG:
        print(msg, flush=True)

# ---------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------

@dataclass
class Vertex:
    pos: np.ndarray
    head: np.ndarray
    parent: int
    cost: float


@dataclass
class PathResult:
    waypoints: List[Vertex]
    cost: float


# ---------------------------------------------------------------------------
# obstacle class
# ---------------------------------------------------------------------------

class Obstacle(Protocol):
    r_c: float  # minimal admissible distance

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """Return distance (mm) from each point to this obstacle."""
        ...


@dataclass
class SphericalObstacle:
    center: np.ndarray
    radius: float  # r_c (mm)

    @property
    def r_c(self) -> float:
        return self.radius

    def distance(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float)
        if pts.ndim == 1:
            pts = pts[None, :]
        diff = pts - self.center[None, :]
        return np.linalg.norm(diff, axis=1)


class VoxelObstacle:
    """
    Obstacle defined by a 3D binary mask from a NIfTI segmentation.
    Coordinates are in *voxel index* space; distances are in mm (via sampling).
    """

    def __init__(self, mask: np.ndarray, voxel_size_mm: np.ndarray, r_c_mm: float, name: str = ""):
        """
        mask: 3D bool or 0/1 array, shape (nx, ny, nz)
        voxel_size_mm: [sx, sy, sz] in mm
        r_c_mm: minimum admissible distance (mm)
        """
        log(f"[VoxelObstacle] Initializing obstacle '{name}'")
        self.name = name
        self.r_c = float(r_c_mm)
        self.mask = mask.astype(bool)
        self.shape = np.array(self.mask.shape, dtype=int)
        self.voxel_size_mm = np.asarray(voxel_size_mm, float)

        # Distance outside mask, in mm
        log(f"[VoxelObstacle:{name}] Computing distance transform (may take a bit)...")
        outside = ~self.mask
        dist_outside = distance_transform_edt(outside, sampling=self.voxel_size_mm)
        dist_outside[~outside] = 0.0
        self.dist_mm = dist_outside
        log(f"[VoxelObstacle:{name}] Distance transform done. Shape={self.shape}")

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """Nearest-neighbour sampling of distance field."""
        pts = np.asarray(pts, float)
        if pts.ndim == 1:
            pts = pts[None, :]

        idx = np.rint(pts).astype(int)
        idx = np.clip(idx, [0, 0, 0], self.shape - 1)
        dvals = self.dist_mm[idx[:, 0], idx[:, 1], idx[:, 2]]
        return dvals


class Environment:
    """
    workplace contains: axis-aligned bounds in voxel coordinates
      arbitrary obstacles (voxel- or sphere-based)
      ε_d (mm): maximum distance where obstacle distance affects cost
    """

    def __init__(
        self,
        bounds_min: Sequence[float],
        bounds_max: Sequence[float],
        obstacles: Optional[List[Obstacle]] = None,
        epsilon_d: float = 20.0,
    ):
        log("[Environment] Creating environment")
        self.bounds_min = np.asarray(bounds_min, float)
        self.bounds_max = np.asarray(bounds_max, float)
        self.obstacles: List[Obstacle] = obstacles or []
        self.epsilon_d: float = float(epsilon_d)
        log(f"[Environment] Bounds_min={self.bounds_min}, Bounds_max={self.bounds_max}")
        log(f"[Environment] Number of obstacles: {len(self.obstacles)}, epsilon_d={self.epsilon_d}")

    def in_bounds(self, p: np.ndarray) -> bool:
        return np.all(p >= self.bounds_min) and np.all(p <= self.bounds_max)

    def collision_free_points(self, pts: np.ndarray) -> bool:
        pts = np.asarray(pts, float)
        if pts.ndim == 1:
            pts = pts[None, :]

        if (pts < self.bounds_min).any() or (pts > self.bounds_max).any():
            return False

        for obs in self.obstacles:
            d = obs.distance(pts)
            if np.any(d <= obs.r_c):
                return False
        return True


# ---------------------------------------------------------------------------
# helper math fcts
# ---------------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0.0:
        return v
    return v / n


def kappa(qa: Vertex, qb: Vertex) -> float:
    """
    κ(qa, qb) = 2 || (qa.pos - qb.pos) × qb.head || / ||qa.pos - qb.pos||^2 FROM PAPER
    """
    diff = qa.pos - qb.pos
    denom = np.linalg.norm(diff) ** 2
    if denom == 0.0:
        return 0.0
    cross = np.cross(diff, qb.head)
    num = 2.0 * np.linalg.norm(cross)
    return num / denom


# ---------------------------------------------------------------------------
# lne and curve fcts
# ---------------------------------------------------------------------------

def StraightLine(qa: Vertex, qb: Vertex, step: float) -> List[Vertex]:
    """linear path bw qa and qb w/ interpolated headings"""
    p1, p2 = qa.pos, qb.pos
    d = np.linalg.norm(p2 - p1)
    if d == 0.0:
        return [qa]

    n_steps = max(1, int(math.ceil(d / step)))
    out: List[Vertex] = []
    for i in range(n_steps + 1):
        t = i / n_steps
        pos = (1.0 - t) * p1 + t * p2
        head = normalize((1.0 - t) * qa.head + t * qb.head)
        out.append(Vertex(pos=pos, head=head, parent=-1, cost=0.0))
    return out


def _constant_curvature_interp(
    qa: Vertex,
    qb: Vertex,
    step: float,
    r_min: float,
) -> List[Vertex]:
    """
    heuristic constant-curvature interpolation from qa to qb
    considering the heading of qa.
    """
    p1, p2 = qa.pos, qb.pos
    chord = p2 - p1
    chord_len = np.linalg.norm(chord)
    if chord_len == 0.0:
        return [qa]

    t0 = normalize(qa.head)
    chord_dir = chord / chord_len

    v = chord_dir - np.dot(chord_dir, t0) * t0
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-6:
        return StraightLine(qa, qb, step)
    v /= v_norm

    dot_tc = max(-1.0, min(1.0, float(np.dot(t0, chord_dir))))
    ang = math.acos(dot_tc)
    if ang == 0.0:
        return StraightLine(qa, qb, step)

    if abs(math.sin(ang)) < 1e-6:
        L = chord_len
    else:
        L = chord_len * (ang / math.sin(ang))

    R = L / ang
    if R < r_min:
        return StraightLine(qa, qb, step)

    axis = normalize(np.cross(t0, v))

    def rot(axis_vec: np.ndarray, theta: float) -> np.ndarray:
        ax = axis_vec
        K = np.array(
            [
                [0.0, -ax[2], ax[1]],
                [ax[2], 0.0, -ax[0]],
                [-ax[1], ax[0], 0.0],
            ]
        )
        I = np.eye(3)
        return I + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)

    center = p1 + R * v
    n_steps = max(1, int(math.ceil(L / step)))
    out: List[Vertex] = []
    for i in range(n_steps + 1):
        s = i / n_steps
        theta = -ang * s
        Rm = rot(axis, theta)
        radial = p1 - center
        pos = center + Rm @ radial
        head = normalize(Rm @ t0)
        out.append(Vertex(pos=pos, head=head, parent=-1, cost=0.0))

    out[-1].pos = p2.copy()
    out[-1].head = normalize(out[-1].head)
    return out


def Curve(qa: Vertex, qb: Vertex, step: float, r_min: float) -> List[Vertex]:
    """curve(qa, qb) with const curvature based on qa.head"""
    return _constant_curvature_interp(qa, qb, step, r_min)


# ---------------------------------------------------------------------------
# edge cost c(E) = a1 f_L + a2 f_D 
# FROM PAPER
# ---------------------------------------------------------------------------

def edge_cost(
    edge: List[Vertex],
    env: Environment,
    L_min: float,
    a1: float,
    a2: float,
) -> float:
    """
    c(E) = a1 f_L + a2 f_D
    distance d_ij are in mm from each obstacle's distance field.
    """
    if len(edge) < 2:
        return 0.0

    # path length in voxel units
    L_vox = sum(float(np.linalg.norm(edge[i + 1].pos - edge[i].pos))
                for i in range(len(edge) - 1))
    f_L = L_vox / L_min if L_min > 0.0 else L_vox

    obstacles = env.obstacles
    N_o = len(obstacles)
    pts = np.vstack([v.pos for v in edge])
    N_e = pts.shape[0]

    if N_o == 0 or N_e == 0:
        f_D = 0.0
    else:
        eps_d = env.epsilon_d
        total = 0.0
        for obs in obstacles:
            rc = obs.r_c
            dist = obs.distance(pts)      # mm
            d_ij = np.clip(dist, rc + 1e-6, eps_d)
            total += np.sum((eps_d - rc) / (d_ij - rc))
        f_D = total / (N_o * N_e)

    return a1 * f_L + a2 * f_D


# ---------------------------------------------------------------------------
# collision chck fct with (qa, qb)
# ---------------------------------------------------------------------------

def CollisionFree(
    env: Environment,
    qa: Vertex,
    qb: Vertex,
    step: float,
    r_min: float,
) -> bool:
    """
    CollisionFree(qa, qb):
      - builds Curve(qa, qb)
      - checks dij > r_c for all samples & obstacles via env.collision_free_points
    """
    edge_pts = Curve(qa, qb, step, r_min)
    if not edge_pts:
        return False
    pos = np.vstack([v.pos for v in edge_pts])
    return env.collision_free_points(pos)

# ---------------------------------------------------------------------------
# load and resample .nii file
# ---------------------------------------------------------------------------

from scipy.ndimage import distance_transform_edt
# you can now delete the unused "zoom" import

def load_mask_native(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI at its native resolution (no resampling) and return:
      - mask: boolean 3D array
      - voxel_size_mm: [sx, sy, sz] in mm from the affine
    """
    log(f"[load_mask_native] Loading NIfTI (no resampling): {path}")
    nii = nib.load(path)
    data = nii.get_fdata()
    affine = nii.affine

    # voxel spacing from affine
    voxel_size_mm = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    mask = data > 0.5

    log(f"[load_mask_native] shape={mask.shape}, voxel_size_mm={voxel_size_mm}")
    return mask, voxel_size_mm


# ---------------------------------------------------------------------------
# build environment from .nii files
# ---------------------------------------------------------------------------

def build_dbs_env_and_masks(
    brain_path: str,
    stn_path: str,
    cc_path: str,
    sulci_path: str,
    vent_path: str,
    r_c_mm: float = 2.0,
    epsilon_d_mm: float = 10.0,
):
    """
    - resamples all masks to 2mm.
    - resamples: center of STN.
    - obstacles: CC, SULCI, VENT.
    - BRAIN_MASK is used for entry surface + visualization.
    """
    log("[build_dbs_env_and_masks] Starting...")
    brain_mask, vox = load_mask_native(brain_path)
    stn_mask, _   = load_mask_native(stn_path)
    cc_mask, _    = load_mask_native(cc_path)
    sulci_mask, _ = load_mask_native(sulci_path)
    vent_mask, _  = load_mask_native(vent_path)

    log("[build_dbs_env_and_masks] All masks loaded & resampled.")
    log(f"  brain_mask.shape={brain_mask.shape}")
    log(f"  stn_mask.shape  ={stn_mask.shape}")
    log(f"  cc_mask.shape   ={cc_mask.shape}")
    log(f"  sulci_mask.shape={sulci_mask.shape}")
    log(f"  vent_mask.shape ={vent_mask.shape}")

    # All shapes must match
    assert brain_mask.shape == stn_mask.shape == cc_mask.shape == sulci_mask.shape == vent_mask.shape, \
        "All resampled masks must have identical shapes"

    nx, ny, nz = brain_mask.shape
    bounds_min = np.array([0.0, 0.0, 0.0])
    bounds_max = np.array([nx - 1.0, ny - 1.0, nz - 1.0])
    log(f"[build_dbs_env_and_masks] Volume bounds: {bounds_min} to {bounds_max}")

    # Goal = STN center of mass (in voxel coords)
    stn_idx = np.argwhere(stn_mask)
    goal_pos = stn_idx[np.random.randint(len(stn_idx))]
    log(f"[build_dbs_env_and_masks] STN center of mass (goal_pos)={goal_pos}")

    if IGNORE_OBSTACLES:
        log("[build_dbs_env_and_masks] IGNORE_OBSTACLES=True – building environment with NO obstacles.")
        obstacles: list[Obstacle] = []
    else:
        obstacles: list[Obstacle] = [
            VoxelObstacle(cc_mask,    vox, r_c_mm, "corpus_callosum"),
            VoxelObstacle(sulci_mask, vox, r_c_mm, "sulci"),
            VoxelObstacle(vent_mask,  vox, r_c_mm, "ventricles"),
        ]


    env = Environment(bounds_min, bounds_max, obstacles=obstacles,
                      epsilon_d=epsilon_d_mm)

    q_goal = Vertex(
        pos=goal_pos.astype(float),
        head=normalize(np.array([0.0, 0.0, 1.0])),
        parent=0,
        cost=0.0,
    )

    log("[build_dbs_env_and_masks] Finished building environment & goal vertex.")
    return env, q_goal, brain_mask, stn_mask, cc_mask, sulci_mask, vent_mask, vox

# ---------------------------------------------------------------------------
# preview brain mask
# ---------------------------------------------------------------------------


def preview_masks_and_entries(
    brain_mask: np.ndarray,
    stn_mask: np.ndarray,
    cc_mask: np.ndarray,
    sulci_mask: np.ndarray,
    vent_mask: np.ndarray,
    entries: list[Vertex],
    voxel_size_mm: np.ndarray,
):
    """
    check:
      - brain surface (physical space)
      - STN + obstacles (light)
      - entry points as spheres on brain surface
    """
    log("[preview_masks_and_entries] Building preview actors...")
    actors = []

    def to_world(p: np.ndarray) -> np.ndarray:
        # voxel -> mm
        return p * voxel_size_mm

    # --- outer surface brain ---
    brain_vol = Volume(brain_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
    brain_surf = brain_vol.isosurface(0.5).smooth(15)
    brain_surf.c("lightgray").alpha(0.25)
    actors.append(brain_surf)

    # --- stn vol (goal) ---
    if stn_mask is not None:
        stn_vol = Volume(stn_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
        stn_surf = stn_vol.isosurface(0.5).smooth(8)
        stn_surf.c("green").alpha(0.9)
        actors.append(stn_surf)

    # --- obstacles ---
    if cc_mask is not None:
        cc_vol = Volume(cc_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
        cc_surf = cc_vol.isosurface(0.5).smooth(8)
        cc_surf.c("cyan").alpha(0.001)
        actors.append(cc_surf)

    if sulci_mask is not None:
        sulci_vol = Volume(sulci_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
        sulci_surf = sulci_vol.isosurface(0.5).decimate(0.8).smooth(8)
        sulci_surf.c("orange").alpha(0.001)
        actors.append(sulci_surf)

    if vent_mask is not None:
        vent_vol = Volume(vent_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
        vent_surf = vent_vol.isosurface(0.5).smooth(8)
        vent_surf.c("blue").alpha(0.001)
        actors.append(vent_surf)

    # --- entry points ---
    if entries:
        entry_pts_world = [to_world(v.pos) for v in entries]
        entry_points_actor = Points(entry_pts_world, r=6)
        entry_points_actor.c("magenta").alpha(0.9)
        actors.append(entry_points_actor)
        log(f"[preview_masks_and_entries] Previewing {len(entries)} entry points.")
    else:
        log("[preview_masks_and_entries] No entries to preview.")

    log("[preview_masks_and_entries] Showing vedo window...")
    show(actors, "Mask & entry sanity check", axes=1, viewup="z")


# ---------------------------------------------------------------------------
# get brain mask
# ---------------------------------------------------------------------------

def sample_surface_entries_on_brain(
    brain_mask: np.ndarray,
    q_goal: Vertex,
    n_entries: int = 50,
) -> list[Vertex]:
    log("[sample_surface_entries_on_brain] Building brain surface mesh...")

    vol = Volume(brain_mask.astype(np.uint8))
    brain_surf = vol.isosurface(0.5).decimate(0.9)

    pts = np.asarray(brain_surf.points)
    log(f"[sample_surface_entries_on_brain] Surface has {pts.shape[0]} points before subsampling.")

    if pts.shape[0] == 0:
        raise RuntimeError("Brain surface mesh has no points")

    n = min(n_entries, pts.shape[0])
    idx = np.random.choice(pts.shape[0], size=n, replace=False)
    pts_sel = pts[idx]
    log(f"[sample_surface_entries_on_brain] Selected {n} entry points on surface.")

    entries: list[Vertex] = []
    for p in pts_sel:
        to_goal = normalize(q_goal.pos - p)

        # add a small lateral component so heading is not perfectly colinear
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(to_goal, tmp)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])

        lateral = normalize(np.cross(to_goal, tmp))
        head = normalize(to_goal + 0.2 * lateral)  # 0.2 = small tilt

        entries.append(Vertex(pos=p.astype(float), head=head, parent=0, cost=0.0))

    log("[sample_surface_entries_on_brain] Finished generating entry vertices.")
    return entries



# ---------------------------------------------------------------------------
# RRT* planner (algorithm 1 – RRTStarPath)
# ---------------------------------------------------------------------------

class RRTStarPlanner:
    def __init__(
        self,
        env: Environment,
        eta: float,
        r_min: float,
        n_samples: int,
        goal_radius: float,
        L_min: float,
        a1: float = 0.5,
        a2: float = 0.5,
        w_d: float = 0.5,
        w_theta: float = 0.5,
        curve_step: float = 1.0,
        goal_mask: Optional[np.ndarray] = None,
    ):
        self.env = env
        self.eta = float(eta)
        self.r_min = float(r_min)
        self.n_samples = int(n_samples)
        self.goal_radius = float(goal_radius)
        self.curve_step = float(curve_step)

        self.L_min = float(L_min)
        self.a1 = float(a1)
        self.a2 = float(a2)

        self.w_d = float(w_d)
        self.w_theta = float(w_theta)
        if abs(self.w_d + self.w_theta - 1.0) > 1e-6:
            s = self.w_d + self.w_theta
            self.w_d /= s
            self.w_theta /= s

        self.d_max = float(np.linalg.norm(self.env.bounds_max - self.env.bounds_min))

        self.vertices: List[Vertex] = []
        self.edges: List[Tuple[int, int]] = []

        self.goal_mask = goal_mask
        log(f"[RRTStarPlanner.__init__] n_samples={self.n_samples}, r_min={self.r_min}, eta={self.eta}, goal_mask_set={self.goal_mask is not None}")

    # -- helpers matching paper's terminology -----------------------------

    def SampleFree(self) -> Vertex:
        mins, maxs = self.env.bounds_min, self.env.bounds_max
        pos = np.array(
            [random.uniform(a, b) for a, b in zip(mins, maxs)], dtype=float
        )
        head = normalize(np.random.normal(size=3))
        return Vertex(pos=pos, head=head, parent=-1, cost=0.0)

    def _rho(self, qa: Vertex, qb: Vertex) -> float:
        d = np.linalg.norm(qa.pos - qb.pos)
        if self.d_max > 0.0:
            d /= self.d_max
        cos_th = float(np.dot(normalize(qa.head), normalize(qb.head)))
        cos_th = max(-1.0, min(1.0, cos_th))
        return self.w_d * (d ** 2) + self.w_theta * ((1.0 - abs(cos_th)) ** 2)

    def NearestReachable(self, q: Vertex) -> Optional[int]:
        """
        NearestReachable(T, q): vertex minimizing ρ and with κ(v, q) < 1/r_min.
        """
        best_idx = None
        best_rho = float("inf")
        for i, v in enumerate(self.vertices):
            if kappa(v, q) >= 1.0 / self.r_min:
                continue
            val = self._rho(v, q)
            if val < best_rho:
                best_rho = val
                best_idx = i
        return best_idx

    def Steer(self, qa: Vertex, qb: Vertex) -> Vertex:
        direction = qb.pos - qa.pos
        d = np.linalg.norm(direction)
        if d == 0.0:
            return qa
        direction_unit = direction / d
        step = min(self.eta, d)
        pos = qa.pos + step * direction_unit
        head = normalize(direction_unit)
        return Vertex(pos=pos, head=head, parent=-1, cost=0.0)

    def Near(self, q: Vertex, r: float) -> List[int]:
        out: List[int] = []
        for i, v in enumerate(self.vertices):
            if np.linalg.norm(v.pos - q.pos) <= r:
                out.append(i)
        return out

    def Cost(self, idx: int) -> float:
        return self.vertices[idx].cost

    def Parent(self, idx: int) -> int:
        return self.vertices[idx].parent

    def Path(self, goal_idx: int) -> List[Vertex]:
        path: List[Vertex] = []
        idx = goal_idx
        while True:
            v = self.vertices[idx]
            path.append(v)
            if v.parent == idx:
                break
            idx = v.parent
        path.reverse()
        return path

    def goal_reached(self, q: Vertex, q_goal: Vertex) -> bool:
        """
        Goal is reached only if the vertex is INSIDE the STN mask.
        """
        if self.goal_mask is not None:
            idx = np.rint(q.pos).astype(int)
            if np.any(idx < 0) or np.any(idx >= self.goal_mask.shape):
                return False
            return bool(self.goal_mask[idx[0], idx[1], idx[2]])

        # If no goal_mask: fall back to spherical target
        return np.linalg.norm(q.pos - q_goal.pos) <= self.goal_radius


        # fallback: spherical neighborhood around STN center? --> or maybe we can change to elliipsoid
       # return np.linalg.norm(q.pos - q_goal.pos) <= self.goal_radius


    # -- main RRT* loop ---------------------------------------------------

    def RRTStarPath(self, q_init: Vertex, q_goal: Vertex) -> Optional[PathResult]:
        log("[RRTStarPlanner.RRTStarPath] Starting RRT* search...")
        self.vertices = []
        self.edges = []

        q0 = Vertex(
            pos=q_init.pos.copy(),
            head=normalize(q_init.head.copy()),
            parent=0,
            cost=0.0,
        )
        self.vertices.append(q0)

        goal_indices: List[int] = []

        for i in range(self.n_samples):

            # print every 100 iterations OR anytime a goal has been found
            if (i % 100 == 0) or bool(goal_indices):
                log(f"[RRTStarPlanner.RRTStarPath] Iteration {i}/{self.n_samples}, vertices={len(self.vertices)}, goal_found={bool(goal_indices)}")
            if goal_indices:
                break

            q_rand = self.SampleFree()

            idx_near = self.NearestReachable(q_rand)
            if idx_near is None:
                continue

            q_near = self.vertices[idx_near]
            q_new = self.Steer(q_near, q_rand)

            if kappa(q_near, q_new) >= 1.0 / self.r_min:
                continue

            if not CollisionFree(self.env, q_near, q_new, self.curve_step, self.r_min):
                continue

            # neighborhood radius r_n (standard RRT* heuristic)
            n = len(self.vertices)
            dim = 3
            gamma_rrt = 2.0 * self.d_max
            r_n = min(
                gamma_rrt * ((math.log(n + 1) / (n + 1)) ** (1.0 / dim)),
                self.eta * 10.0,
            )
            Q_near_idx = self.Near(q_new, r_n)

            # choose best parent
            q_min_parent = idx_near
            E_min = Curve(q_near, q_new, self.curve_step, self.r_min)
            c_min = self.Cost(idx_near) + edge_cost(
                E_min, self.env, self.L_min, self.a1, self.a2
            )

            for j in Q_near_idx:
                q_near2 = self.vertices[j]
                if kappa(q_near2, q_new) >= 1.0 / self.r_min:
                    continue
                if not CollisionFree(
                    self.env, q_near2, q_new, self.curve_step, self.r_min
                ):
                    continue
                E_candidate = Curve(q_near2, q_new, self.curve_step, self.r_min)
                if not E_candidate:
                    continue
                new_cost = self.Cost(j) + edge_cost(
                    E_candidate, self.env, self.L_min, self.a1, self.a2
                )
                if new_cost < c_min:
                    q_min_parent = j
                    c_min = new_cost
                    E_min = E_candidate

            new_idx = len(self.vertices)
            q_new.parent = q_min_parent
            q_new.cost = c_min
            self.vertices.append(q_new)
            self.edges.append((q_min_parent, new_idx))

            # rewire
            for j in Q_near_idx:
                if j == new_idx:
                    continue
                q_near2 = self.vertices[j]
                if kappa(q_new, q_near2) >= 1.0 / self.r_min:
                    continue
                if not CollisionFree(
                    self.env, q_new, q_near2, self.curve_step, self.r_min
                ):
                    continue
                E_rev = Curve(q_new, q_near2, self.curve_step, self.r_min)
                new_cost = q_new.cost + edge_cost(
                    E_rev, self.env, self.L_min, self.a1, self.a2
                )
                if new_cost < self.Cost(j):
                    old_parent = self.Parent(j)
                    self.vertices[j].parent = new_idx
                    self.vertices[j].cost = new_cost
                    try:
                        self.edges.remove((old_parent, j))
                    except ValueError:
                        pass
                    self.edges.append((new_idx, j))

            if self.goal_reached(q_new, q_goal):
                log(f"[RRTStarPlanner.RRTStarPath] Goal region reached at iteration {i} with vertex index {new_idx}")
                goal_indices.append(new_idx)

        if not goal_indices:
            log("[RRTStarPlanner.RRTStarPath] No goal reached after all samples.")
            return None

        best_idx = min(goal_indices, key=lambda idx: self.vertices[idx].cost)
        path_vertices = self.Path(best_idx)
        total_cost = edge_cost(path_vertices, self.env, self.L_min, self.a1, self.a2)
        log(f"[RRTStarPlanner.RRTStarPath] Finished. Best path cost={total_cost:.4f}, length={len(path_vertices)}")
        return PathResult(waypoints=path_vertices, cost=total_cost)


# ---------------------------------------------------------------------------
# Algorithm-1 with: LinearPath, CurvePath, MinimumCost
# ---------------------------------------------------------------------------

def LinearPath(
    env: Environment,
    q_init: Vertex,
    q_goal: Vertex,
    step: float,
    L_min: float,
    a1: float,
    a2: float,
) -> Optional[PathResult]:
    log("[LinearPath] Attempting straight path...")
    E = StraightLine(q_init, q_goal, step)
    pos = np.vstack([v.pos for v in E])
    if env.collision_free_points(pos):
        c = edge_cost(E, env, L_min, a1, a2)
        log(f"[LinearPath] Straight path is collision-free. Cost={c:.4f}")
        log(f"[LinearPath] L_min={L_min:.6f}, cost={c:.6f}")
        return PathResult(waypoints=E, cost=c)
    log("[LinearPath] Straight path is in collision.")
    log(f"[LinearPath] L_min={L_min:.6f}, cost=N/A (path invalid)")
    return None


def CurvePath(
    env: Environment,
    q_init: Vertex,
    q_goal: Vertex,
    step: float,
    r_min: float,
    L_min: float,
    a1: float,
    a2: float,
) -> Optional[PathResult]:
    log("[CurvePath] Attempting one-shot curved path...")
    E = Curve(q_init, q_goal, step, r_min)
    if not E:
        log("[CurvePath] Curve interpolation returned empty path.")
        return None
    pos = np.vstack([v.pos for v in E])
    if env.collision_free_points(pos) and kappa(q_init, q_goal) < 1.0 / r_min:
        c = edge_cost(E, env, L_min, a1, a2)
        log(f"[CurvePath] Curved path is collision-free. Cost={c:.4f}")
        log(f"[CurvePath] L_min={L_min:.6f}, cost={c:.6f}")
        return PathResult(waypoints=E, cost=c)
    log("[CurvePath] Curved path is in collision or violates curvature constraint.")
    log(f"[CurvePath] L_min={L_min:.6f}, cost=N/A (path invalid)")


    return None


def MinimumCost(paths: List[PathResult]) -> Optional[PathResult]:
    if not paths:
        log("[MinimumCost] No candidate paths provided.")
        return None
    best = min(paths, key=lambda p: p.cost)
    log(f"[MinimumCost] Selected path with cost={best.cost:.4f} from {len(paths)} candidates.")
    return best


def plan_flexible_needle_path(
    env: Environment,
    q_init: Vertex,
    q_goal: Vertex,
    r_min: float,
    step: float = 1.0,
    n_samples: int = 2000,
    goal_radius: float = 2.0,
    a1: float = 0.5,
    a2: float = 0.5,
    w_d: float = 0.5,
    w_theta: float = 0.5,
    goal_mask: Optional[np.ndarray] = None,   # <--- NEW
) -> Optional[PathResult]:
    """
    full alg 1:
      1: P ← LinearPath(q_init, q_goal) ∪ CurvePath(q_init, q_goal)
      2: P ← P ∪ RRTStarPath(q_init, q_goal, N)
      3: P_opt ← MinimumCost(P)
    """
    L_min = float(np.linalg.norm(q_goal.pos - q_init.pos))

    candidate_paths: List[PathResult] = []

    lp = LinearPath(env, q_init, q_goal, step, L_min, a1, a2)
    if lp is not None:
        candidate_paths.append(lp)

    cp = CurvePath(env, q_init, q_goal, step, r_min, L_min, a1, a2)
    if cp is not None:
        candidate_paths.append(cp)

    planner = RRTStarPlanner(
        env=env,
        eta=step,
        r_min=r_min,
        n_samples=n_samples,
        goal_radius=goal_radius,
        L_min=L_min,
        a1=a1,
        a2=a2,
        w_d=w_d,
        w_theta=w_theta,
        curve_step=step / 2.0,
        goal_mask=goal_mask,
    )
    rrt_path = planner.RRTStarPath(q_init, q_goal)
    if rrt_path is not None:
        candidate_paths.append(rrt_path)

    return MinimumCost(candidate_paths)

# ---------------------------------------------------------------------------
# tiny example to sanity-check
#### IGNORE 
# ---------------------------------------------------------------------------

def _example():
    log("[_example] Running toy example (not used in main optimize_dbs).")
    obs = SphericalObstacle(center=np.array([50.0, 50.0, 50.0]), radius=15.0)
    env = Environment(
        bounds_min=[0.0, 0.0, 0.0],
        bounds_max=[100.0, 100.0, 100.0],
        obstacles=[obs],
        epsilon_d=40.0,
    )

    q_init = Vertex(
        pos=np.array([10.0, 10.0, 10.0]),
        head=normalize(np.array([1.0, 0.2, 0.1])),
        parent=0,
        cost=0.0,
    )
    q_goal = Vertex(
        pos=np.array([90.0, 90.0, 90.0]),
        head=normalize(np.array([-0.5, -1.0, -0.3])),
        parent=0,
        cost=0.0,
    )

    r_min = 10.0
    result = plan_flexible_needle_path(
        env,
        q_init,
        q_goal,
        r_min=r_min,
        step=3.0,
        n_samples=3000,
        goal_radius=5.0,
        a1=0.5,
        a2=0.5,
        w_d=0.7,
        w_theta=0.3,
    )

    if result is None:
        log("[_example] No path found.")
    else:
        log(f"[_example] Found path with {len(result.waypoints)} vertices, cost={result.cost:.4f}")
        for i, v in enumerate(result.waypoints[:10]):
            print(f"{i:2d}: pos={v.pos.round(2)}, head={v.head.round(2)}")


########################### OPTIMIZATION FUNCTION ###########################

# ---------------------------------------------------------------------------
# gen entry points
# ---------------------------------------------------------------------------

def generate_entry_grid(env: Environment, q_goal: Vertex,
                        nx: int = 8, ny: int = 8) -> List[Vertex]:
    """
    Sample candidate entry points on the top (superior) plane of the volume.
    Coordinates are in voxel space.
    """
    log("[generate_entry_grid] Generating grid of entry points on top plane...")
    xs = np.linspace(env.bounds_min[0] + 1, env.bounds_max[0] - 1, nx)
    ys = np.linspace(env.bounds_min[1] + 1, env.bounds_max[1] - 1, ny)
    z  = env.bounds_max[2]

    entries: List[Vertex] = []
    for x in xs:
        for y in ys:
            pos = np.array([x, y, z], float)
            if not env.in_bounds(pos):
                continue
            # heading = roughly towards goal
            head = normalize(q_goal.pos - pos)
            entries.append(Vertex(pos=pos, head=head, parent=0, cost=0.0))
    log(f"[generate_entry_grid] Generated {len(entries)} entry points.")
    return entries

# ---------------------------------------------------------------------------
# optimization fct
# ---------------------------------------------------------------------------

def optimize_over_entries(
    env: Environment,
    q_goal: Vertex,
    entries: list[Vertex],
    r_min: float,
    step: float = 1.0,
    n_samples: int = 2000,
    goal_radius: float = 2.0,
    a1: float = 0.1,
    a2: float = 0.9,
    w_d: float = 0.5,
    w_theta: float = 0.5,
    goal_mask: Optional[np.ndarray] = None,
) -> tuple[Optional[PathResult], list[tuple[Vertex, PathResult]], list[Vertex]]:

    all_successes: list[tuple[Vertex, PathResult]] = []
    all_failures: list[Vertex] = []

    # Run planner on each entry
    for q_init in entries:
        res = plan_flexible_needle_path(
            env=env,
            q_init=q_init,
            q_goal=q_goal,
            r_min=r_min,
            step=step,
            n_samples=n_samples,
            goal_radius=goal_radius,
            a1=a1,
            a2=a2,
            w_d=w_d,
            w_theta=w_theta,
            goal_mask=goal_mask,
        )

        if res is not None:
            all_successes.append((q_init, res))
        else:
            all_failures.append(q_init)

    # Nothing succeeded → return all 3 values
    if not all_successes:
        return None, all_successes, all_failures

    # Select minimum-cost success
    best_entry, best_path = min(all_successes, key=lambda qr: qr[1].cost)

    return best_path, all_successes, all_failures



# ---------------------------------------------------------------------------
# plotting fct
# ---------------------------------------------------------------------------

def plot_trajectories_3d_with_structures(
    best: PathResult,
    all_results: list[tuple[Vertex, PathResult]],
    failed_entries: list[Vertex],
    brain_mask: np.ndarray,
    stn_mask: np.ndarray,
    cc_mask: np.ndarray,
    sulci_mask: np.ndarray,
    vent_mask: np.ndarray,
    voxel_size_mm: np.ndarray,
    q_goal: Vertex,
    r_min: float,
    step: float = 1.0,
):
    """
    Visualize:
      - ONE straight path (blue)
      - ONE curved path (orange, with min radius 40 mm)
      - ONE RRT* path (red = 'best')

    All three start at the SAME point on the brain surface.
    """

    log("[plot_trajectories_3d_with_structures] Building 3D actors for visualization...")
    actors = []

    def to_world(p: np.ndarray) -> np.ndarray:
        # voxel -> mm
        return p * voxel_size_mm

    # --- structural surfaces ------------------------------------------------
    log("[plot_trajectories_3d_with_structures] Generating brain surface...")
    brain_vol = Volume(brain_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
    brain_surf = brain_vol.isosurface(0.5).smooth(20)
    brain_surf.c("lightgray").alpha(0.25)
    actors.append(brain_surf)

    log("[plot_trajectories_3d_with_structures] Generating corpus callosum surface...")
    cc_vol = Volume(cc_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
    cc_surf = cc_vol.isosurface(0.5).smooth(10)
    cc_surf.c("cyan").alpha(0.001)
    actors.append(cc_surf)

    log("[plot_trajectories_3d_with_structures] Generating sulci surface...")
    sulci_vol = Volume(sulci_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
    sulci_surf = sulci_vol.isosurface(0.5).decimate(0.8).smooth(10)
    sulci_surf.c("orange").alpha(0.001)
    actors.append(sulci_surf)

    log("[plot_trajectories_3d_with_structures] Generating ventricles surface...")
    vent_vol = Volume(vent_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
    vent_surf = vent_vol.isosurface(0.5).smooth(10)
    vent_surf.c("blue").alpha(0.001)
    actors.append(vent_surf)

    log("[plot_trajectories_3d_with_structures] Generating STN surface...")
    stn_vol = Volume(stn_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
    stn_surf = stn_vol.isosurface(0.5).smooth(8)
    stn_surf.c("green").alpha(0.9)
    actors.append(stn_surf)

    # ------------------------------------------------------------------
    # Find the entry vertex corresponding to the chosen best path
    # ------------------------------------------------------------------
    log("[plot_trajectories_3d_with_structures] Finding best entry for chosen path...")
    best_entry: Optional[Vertex] = None
    for q_init, path in all_results:
        if path is best:
            best_entry = q_init
            break

    if best_entry is None:
        if all_results:
            best_entry = all_results[0][0]
            log("[plot_trajectories_3d_with_structures] WARNING: best path not found in all_results; using first entry.")
        else:
            v0 = best.waypoints[0]
            best_entry = Vertex(pos=v0.pos.copy(), head=v0.head.copy(), parent=0, cost=0.0)
            log("[plot_trajectories_3d_with_structures] WARNING: all_results empty; using best path start as entry.")

    # Common entry point in WORLD space, snapped to brain surface
    entry_world_guess = to_world(best_entry.pos)
    entry_world = np.array(brain_surf.closest_point(entry_world_guess))

    # Goal in WORLD space (STN)
    goal_world = to_world(best.waypoints[-1].pos)

    # Mark entry + goal
    entry_actor = Points([entry_world], r=8)
    entry_actor.c("white").alpha(1.0)
    actors.append(entry_actor)

    goal_actor = Points([goal_world], r=10)
    goal_actor.c("yellow").alpha(1.0)
    actors.append(goal_actor)

    # ------------------------------------------------------------------
    # 1) STRAIGHT PATH (blue) – from entry to STN
    # ------------------------------------------------------------------
    log("[plot_trajectories_3d_with_structures] Adding straight path (blue)...")

    straight_edge = StraightLine(best_entry, q_goal, step)
    straight_pts_world = np.array([to_world(v.pos) for v in straight_edge])

    # force same start point
    straight_pts_world[0] = entry_world

    straight_actor = Line(straight_pts_world)
    straight_actor.c("blue").alpha(0.9).lw(3)
    actors.append(straight_actor)

    # ------------------------------------------------------------------
    # 2) CURVED PATH (orange) – cubic Bézier with curvature limit
    # ------------------------------------------------------------------
    log("[plot_trajectories_3d_with_structures] Adding curved path (orange) with curvature limit...")

    def estimate_min_radius(points: np.ndarray) -> float:
        """
        Estimate minimum radius of curvature along a polyline.
        Returns +inf if we can't compute curvature.
        """
        if points.shape[0] < 3:
            return float("inf")

        min_R = float("inf")
        for i in range(1, points.shape[0] - 1):
            A = points[i - 1]
            B = points[i]
            C = points[i + 1]

            AB = B - A
            BC = C - B
            AC = C - A

            a = np.linalg.norm(AB)
            b = np.linalg.norm(BC)
            c = np.linalg.norm(AC)
            if a == 0.0 or b == 0.0 or c == 0.0:
                continue

            # curvature k = |(B-A) x (C-A)| / (|AB| * |BC| * |AC|)
            cross = np.cross(AB, AC)
            area2 = np.linalg.norm(cross)  # = 2 * area of triangle
            k = area2 / (a * b * c)
            if k == 0.0:
                continue
            R = 1.0 / k
            if R < min_R:
                min_R = R
        return min_R

    def bezier_curve_with_limit(
        p0: np.ndarray,
        p3: np.ndarray,
        R_min_mm: float = 40.0,
        bend_scale_init: float = 0.35,
        n_points: int = 120,
        max_iter: int = 12,
    ) -> np.ndarray:
        """
        Cubic Bézier from p0 to p3 with lateral offset, but enforce
        minimum radius of curvature R_min_mm by shrinking bend_scale
        if needed.
        """
        chord = p3 - p0
        chord_len = np.linalg.norm(chord)
        if chord_len == 0.0:
            return np.repeat(p0[None, :], n_points, axis=0)

        d = chord / chord_len

        # choose "up" not parallel to d
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(d, up)) > 0.9:
            up = np.array([0.0, 1.0, 0.0])

        lateral = np.cross(d, up)
        lateral /= np.linalg.norm(lateral)

        bend_scale = bend_scale_init
        pts = None

        for _ in range(max_iter):
            offset = bend_scale * chord_len * lateral

            p1 = p0 + chord * 0.33 + offset
            p2 = p0 + chord * 0.66 + offset

            ts = np.linspace(0.0, 1.0, n_points)
            pts = (
                (1 - ts)[:, None] ** 3 * p0[None, :]
                + 3 * (1 - ts)[:, None] ** 2 * ts[:, None] * p1[None, :]
                + 3 * (1 - ts)[:, None] * ts[:, None] ** 2 * p2[None, :]
                + ts[:, None] ** 3 * p3[None, :]
            )

            R_min_est = estimate_min_radius(pts)
            log(f"[bezier_curve_with_limit] bend_scale={bend_scale:.4f}, estimated min R={R_min_est:.2f} mm")

            # If we cannot compute curvature, or we're within limits, accept.
            if not np.isfinite(R_min_est) or R_min_est >= R_min_mm:
                break

            # Too tight: reduce bend and try again.
            bend_scale *= 0.5

        return pts

    curve_pts_world = bezier_curve_with_limit(
        entry_world,
        goal_world,
        R_min_mm=100.0,      # <- your 40 mm minimum radius
        bend_scale_init=0.35,
        n_points=120,
    )
    curve_actor = Line(curve_pts_world)
    curve_actor.c("orange").alpha(0.9).lw(3)
    actors.append(curve_actor)

    # ------------------------------------------------------------------
    # 3) RRT* PATH (red) – 'best' path
    # ------------------------------------------------------------------
    log("[plot_trajectories_3d_with_structures] Adding RRT* path (red)...")
    rrt_pts_world = np.array([to_world(v.pos) for v in best.waypoints])

    # force same entry for visual comparison
    rrt_pts_world[0] = entry_world

    rrt_actor = Line(rrt_pts_world)
    rrt_actor.c("red").alpha(1.0).lw(4)
    actors.append(rrt_actor)

    log("[plot_trajectories_3d_with_structures] Rendering with vedo.show()...")
    show(actors, "DBS trajectories (straight / curved / RRT*)", axes=1, viewup="z")



def find_first_collision_point(env: Environment, edge: list[Vertex]) -> tuple[Optional[np.ndarray], Optional[Obstacle]]:
    """
    Given a sequence of vertices (a path), return the first point along the path
    that collides with any obstacle (d <= r_c), and the corresponding obstacle.
    Points are in voxel coordinates.
    """
    if not edge:
        return None, None

    for v in edge:
        p = v.pos
        for obs in env.obstacles:
            d = obs.distance(p)
            # keep in mind? obs.distance returns np.ndarray
            d_val = float(d[0]) if isinstance(d, np.ndarray) else float(d)
            if d_val <= obs.r_c:
                return p, obs
    return None, None

def debug_sample_paths_for_entry(
    env: Environment,
    q_init: Vertex,
    q_goal: Vertex,
    brain_mask: np.ndarray,
    stn_mask: np.ndarray,
    cc_mask: np.ndarray,
    sulci_mask: np.ndarray,
    vent_mask: np.ndarray,
    voxel_size_mm: np.ndarray,
    r_min: float,
    step: float = 1.0,
    n_samples: int = 500,
    goal_radius: float = 2.0,
    a1: float = 0.1,
    a2: float = 0.9,
    w_d: float = 0.7,
    w_theta: float = 0.3,
):
    """
    for a single entry,
      - compute LINE path
      - compute CURVE path
      - run RRT* once
    and display them all in 3D on top of the anatomy w/ red markers at
    collision points for line/curve.
    """
    log("[debug_sample_paths_for_entry] Building debug sample for one entry...")
    actors = []

    def to_world(p: np.ndarray) -> np.ndarray:
        return p * voxel_size_mm

    # --- anatomy surfaces ---------------------------
    brain_vol = Volume(brain_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
    brain_surf = brain_vol.isosurface(0.5).smooth(15)
    brain_surf.c("lightgray").alpha(0.25)
    actors.append(brain_surf)

    if stn_mask is not None:
        stn_vol = Volume(stn_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
        stn_surf = stn_vol.isosurface(0.5).smooth(8)
        stn_surf.c("green").alpha(0.9)
        actors.append(stn_surf)

    if cc_mask is not None:
        cc_vol = Volume(cc_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
        cc_surf = cc_vol.isosurface(0.5).smooth(8)
        cc_surf.c("cyan").alpha(0.4)
        actors.append(cc_surf)

    if sulci_mask is not None:
        sulci_vol = Volume(sulci_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
        sulci_surf = sulci_vol.isosurface(0.5).decimate(0.8).smooth(8)
        sulci_surf.c("orange").alpha(0.25)
        actors.append(sulci_surf)

    if vent_mask is not None:
        vent_vol = Volume(vent_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
        vent_surf = vent_vol.isosurface(0.5).smooth(8)
        vent_surf.c("blue").alpha(0.6)
        actors.append(vent_surf)

    # --- entry & goal markers ---------------------------------------------
    entry_pt_world = to_world(q_init.pos)
    entry_actor = Points([entry_pt_world], r=10)
    entry_actor.c("magenta").alpha(1.0)
    actors.append(entry_actor)

    goal_pt_world = to_world(q_goal.pos)
    goal_actor = Points([goal_pt_world], r=10)
    goal_actor.c("yellow").alpha(1.0)
    actors.append(goal_actor)

    # --- LINE path --------------------------------------------------------
    L_min = float(np.linalg.norm(q_goal.pos - q_init.pos))

    log("[debug_sample_paths_for_entry] Computing straight line path...")
    line_edge = StraightLine(q_init, q_goal, step)
    line_pts_world = np.array([to_world(v.pos) for v in line_edge])
    line_actor = Line(line_pts_world)
    line_actor.c("blue").alpha(0.8).lw(3)
    actors.append(line_actor)

    line_collision_vox, line_obs = find_first_collision_point(env, line_edge)
    if line_collision_vox is not None:
        log(f"[debug_sample_paths_for_entry] Line path hits obstacle '{getattr(line_obs, 'name', '?')}'")
        col_world = to_world(line_collision_vox)
        col_actor = Points([col_world], r=12)
        col_actor.c("red").alpha(1.0)
        actors.append(col_actor)
    else:
        log("[debug_sample_paths_for_entry] Line path is collision-free.")

    # --- CURVE path -------------------------------------------------------
    log("[debug_sample_paths_for_entry] Computing curved path...")
    curve_edge = Curve(q_init, q_goal, step, r_min)
    curve_pts_world = np.array([to_world(v.pos) for v in curve_edge])
    curve_actor = Line(curve_pts_world)
    curve_actor.c("orange").alpha(0.8).lw(3)
    actors.append(curve_actor)

    curve_collision_vox, curve_obs = find_first_collision_point(env, curve_edge)
    if curve_collision_vox is not None:
        log(f"[debug_sample_paths_for_entry] Curve path hits obstacle '{getattr(curve_obs, 'name', '?')}'")
        col_world = to_world(curve_collision_vox)
        col_actor = Points([col_world], r=12)
        col_actor.c("red").alpha(1.0)
        actors.append(col_actor)
    else:
        log("[debug_sample_paths_for_entry] Curve path is collision-free.")

    # --- RRT* path --------------------------------------------------------
    log("[debug_sample_paths_for_entry] Running RRT* for this entry (sample)...")
    planner = RRTStarPlanner(
        env=env,
        eta=step,
        r_min=r_min,
        n_samples=n_samples,
        goal_radius=goal_radius,
        L_min=L_min,
        a1=a1,
        a2=a2,
        w_d=w_d,
        w_theta=w_theta,
        curve_step=step / 2.0,
        goal_mask=stn_mask,  # goal is STN volume
    )
    rrt_result = planner.RRTStarPath(q_init, q_goal)

    if rrt_result is not None:
        rrt_pts_world = np.array([to_world(v.pos) for v in rrt_result.waypoints])
        rrt_actor = Line(rrt_pts_world)
        rrt_actor.c("magenta").alpha(1.0).lw(4)
        actors.append(rrt_actor)
        log(f"[debug_sample_paths_for_entry] RRT* path found: {len(rrt_result.waypoints)} waypoints, cost={rrt_result.cost:.4f}")
    else:
        log("[debug_sample_paths_for_entry] RRT* did not find a path for this sample.")

    log("[debug_sample_paths_for_entry] Showing debug sample paths (line=blue, curve=orange, RRT*=magenta, collisions=red)...")
    show(actors, "Sample LINE / CURVE / RRT* paths", axes=1, viewup="z")


# ---------------------------------------------------------------------------
# optimization function call
# ---------------------------------------------------------------------------

def optimize_dbs():
    log("[optimize_dbs] ===== Starting DBS optimization pipeline =====")
    env, q_goal, brain_mask, stn_mask, cc_mask, sulci_mask, vent_mask, vox = \
        build_dbs_env_and_masks(
            brain_path="/Users/vivienyu/Desktop/opt_proj/Optimization_Project/FINAL_BRAIN_ATLAS/algorithm_opt/files/brain_mask.nii",
            stn_path="/Users/vivienyu/Desktop/opt_proj/Optimization_Project/FINAL_BRAIN_ATLAS/algorithm_opt/files/subthalamic_nucleus.nii",
            cc_path="/Users/vivienyu/Desktop/opt_proj/Optimization_Project/FINAL_BRAIN_ATLAS/algorithm_opt/files/corpus_callosum.nii",
            sulci_path="/Users/vivienyu/Desktop/opt_proj/Optimization_Project/FINAL_BRAIN_ATLAS/algorithm_opt/files/sulci.nii",
            vent_path="/Users/vivienyu/Desktop/opt_proj/Optimization_Project/FINAL_BRAIN_ATLAS/algorithm_opt/files/ventricles.nii",
            r_c_mm=2.0,
            epsilon_d_mm=20.0,
        )

    voxel_size = vox  # [2.0, 2.0, 2.0]
    r_min_vox = 1  # 1/R = 0.025, R = 40
    N_SURFACE_ENTRIES = 1
    N_ITERATIONS = 200

    log(f"[optimize_dbs] Voxel size (mm) = {voxel_size}")

    # part 1: sample entry points on outer brain surface
    log("[optimize_dbs] Sampling entry points on outer brain surface...")
    entries = sample_surface_entries_on_brain(brain_mask, q_goal, n_entries=N_SURFACE_ENTRIES)
    log(f"[optimize_dbs] Generated {len(entries)} surface entries.")

    # 3D sanity check of brain surface + STN + obstacles + entries
    if PREVIEW_ENTRIES:
        log("[optimize_dbs] Launching preview of masks and entry points...")
        preview_masks_and_entries(
            brain_mask=brain_mask,
            stn_mask=stn_mask,
            cc_mask=cc_mask,
            sulci_mask=sulci_mask,
            vent_mask=vent_mask,
            entries=entries,
            voxel_size_mm=voxel_size,
        )

    # part 2: optimize over generated entries
    log(f"[optimize_dbs] Starting optimization over entries with r_min_vox={r_min_vox}...")
    best, all_successes, all_failures = optimize_over_entries(
        env,
        q_goal,
        entries=entries,
        r_min=r_min_vox,
        step=1.0,
        n_samples=N_ITERATIONS,
        goal_radius=2.0,
        a1=0.1,
        a2=0.9,
        w_d=0.7,
        w_theta=0.3,
        goal_mask=stn_mask,
    )

    if best is None:
        log("[optimize_dbs] No viable path from any entry. Exiting.")
        print("No viable path from any entry.")
        return None, [], voxel_size

    log(f"[optimize_dbs] Best path cost: {best.cost:.4f}, waypoints={len(best.waypoints)}")
    print(f"Best path cost: {best.cost:.4f}, {len(best.waypoints)} waypoints")

    # part 3: directly plot on full anatomy
    log("[optimize_dbs] Plotting trajectories with anatomical structures...")
    plot_trajectories_3d_with_structures(
        best,
        all_successes,
        all_failures,
        brain_mask,
        stn_mask,
        cc_mask,
        sulci_mask,
        vent_mask,
        voxel_size_mm=voxel_size,
        q_goal=q_goal,
        r_min=r_min_vox,
        step=1.0,
    )

    log("[optimize_dbs] ===== Finished DBS optimization pipeline =====")
    return best, all_successes, all_failures, voxel_size


if __name__ == "__main__":
    optimize_dbs()
