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

def log(msg: str) -> None:
    if DEBUG:
        print(msg, flush=True)

# ---------------------------------------------------------------------------
# Basic data types
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
# Generic obstacle interface + voxel obstacles
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
    Workspace with:
      - axis-aligned bounds in voxel coordinates
      - arbitrary obstacles (voxel- or sphere-based)
      - ε_d (mm): maximum distance where obstacle distance affects cost
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
# Helper math
# ---------------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0.0:
        return v
    return v / n


def kappa(qa: Vertex, qb: Vertex) -> float:
    """
    κ(qa, qb) = 2 || (qa.pos - qb.pos) × qb.head || / ||qa.pos - qb.pos||^2
    """
    diff = qa.pos - qb.pos
    denom = np.linalg.norm(diff) ** 2
    if denom == 0.0:
        return 0.0
    cross = np.cross(diff, qb.head)
    num = 2.0 * np.linalg.norm(cross)
    return num / denom


# ---------------------------------------------------------------------------
# Line and constant-curvature curve
# ---------------------------------------------------------------------------

def Line(qa: Vertex, qb: Vertex, step: float) -> List[Vertex]:
    """Straight-line path between qa and qb with interpolated headings."""
    # Not logging every call to avoid spam
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
    Heuristic constant-curvature interpolation from qa to qb
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
        return Line(qa, qb, step)
    v /= v_norm

    dot_tc = max(-1.0, min(1.0, float(np.dot(t0, chord_dir))))
    ang = math.acos(dot_tc)
    if ang == 0.0:
        return Line(qa, qb, step)

    if abs(math.sin(ang)) < 1e-6:
        L = chord_len
    else:
        L = chord_len * (ang / math.sin(ang))

    R = L / ang
    if R < r_min:
        return Line(qa, qb, step)

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
    """Curve(qa, qb) with constant curvature based on qa.head."""
    return _constant_curvature_interp(qa, qb, step, r_min)


# ---------------------------------------------------------------------------
# Edge cost c(E) = a1 f_L + a2 f_D
# ---------------------------------------------------------------------------

def edge_cost(
    edge: List[Vertex],
    env: Environment,
    L_min: float,
    a1: float,
    a2: float,
) -> float:
    """
    c(E) = a1 f_L + a2 f_D with arbitrary-shaped obstacles.
    Distances d_ij are in mm from each obstacle's distance field.
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
# CollisionFree(qa, qb)
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
# Load and Resample .nii file
# ---------------------------------------------------------------------------

def load_and_resample_mask_2mm(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI, resample to 2x2x2 mm^3 using nearest neighbour,
    and return (mask, voxel_size_mm).
    """
    log(f"[load_and_resample_mask_2mm] Loading NIfTI: {path}")
    nii = nib.load(path)
    data = nii.get_fdata()
    affine = nii.affine

    # original spacing (mm)
    orig_spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    new_spacing = np.array([2.0, 2.0, 2.0])

    zoom_factors = orig_spacing / new_spacing
    log(f"[load_and_resample_mask_2mm] Original spacing={orig_spacing}, zoom_factors={zoom_factors}")
    # Nearest-neighbour preserves labels
    resampled = zoom(data, zoom=zoom_factors, order=0)

    # new affine: same orientation, new spacing
    R = affine[:3, :3]
    R_unit = R / orig_spacing
    new_aff = np.eye(4)
    new_aff[:3, :3] = R_unit * new_spacing
    new_aff[:3, 3] = affine[:3, 3]  # keep origin

    mask = resampled > 0.5
    log(f"[load_and_resample_mask_2mm] Finished resampling. New shape={mask.shape}")
    return mask, new_spacing

# ---------------------------------------------------------------------------
# Build DBS environment from .nii files
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
    - Resamples all masks to 2mm.
    - Goal: center of STN.
    - Obstacles: CC, SULCI, VENT.
    - BRAIN_MASK is used for entry surface + visualization.
    """
    log("[build_dbs_env_and_masks] Starting...")
    brain_mask, vox = load_and_resample_mask_2mm(brain_path)
    stn_mask, _   = load_and_resample_mask_2mm(stn_path)
    cc_mask, _    = load_and_resample_mask_2mm(cc_path)
    sulci_mask, _ = load_and_resample_mask_2mm(sulci_path)
    vent_mask, _  = load_and_resample_mask_2mm(vent_path)

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
    goal_pos = stn_idx.mean(axis=0)
    log(f"[build_dbs_env_and_masks] STN center of mass (goal_pos)={goal_pos}")

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
# Generate Brain Mask
# ---------------------------------------------------------------------------

def sample_surface_entries_on_brain(
    brain_mask: np.ndarray,
    q_goal: Vertex,
    n_entries: int = 50,
) -> list[Vertex]:
    """
    Sample candidate entry points *on the outer brain surface*.
    Coordinates are in voxel space (Volume uses voxel indices by default).
    """
    log("[sample_surface_entries_on_brain] Building brain surface mesh...")
    # Build a surface mesh of the brain mask
    vol = Volume(brain_mask.astype(np.uint8))
    brain_surf = vol.isosurface(0.5).decimate(0.9)  # decimate to reduce vertices

    pts = np.asarray(brain_surf.points)  # shape (N,3) in voxel coordinates
    log(f"[sample_surface_entries_on_brain] Surface has {pts.shape[0]} points before subsampling.")

    if pts.shape[0] == 0:
        raise RuntimeError("Brain surface mesh has no points")

    # Randomly subsample surface points
    n = min(n_entries, pts.shape[0])
    idx = np.random.choice(pts.shape[0], size=n, replace=False)
    pts_sel = pts[idx]
    log(f"[sample_surface_entries_on_brain] Selected {n} entry points on surface.")

    entries: list[Vertex] = []
    for p in pts_sel:
        head = normalize(q_goal.pos - p)   # point inward toward STN
        entries.append(Vertex(pos=p.astype(float), head=head, parent=0, cost=0.0))
    log("[sample_surface_entries_on_brain] Finished generating entry vertices.")
    return entries



# ---------------------------------------------------------------------------
# RRT* planner (Algorithm 1 – RRTStarPath)
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
        goal_mask: Optional[np.ndarray] = None,   # <--- NEW
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

        self.goal_mask = goal_mask      # <--- STORE IT
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
        Goal:
          - Primary: point inside / very near the STN (goal_mask)
          - Fallback: within goal_radius of q_goal.pos
        """
        # Volume-based goal using STN mask
        if self.goal_mask is not None:
            idx = np.rint(q.pos).astype(int)

            if np.any(idx < 0) or np.any(idx >= self.goal_mask.shape):
                return False

            i, j, k = idx
            i0 = max(0, i - 1); i1 = min(self.goal_mask.shape[0] - 1, i + 1)
            j0 = max(0, j - 1); j1 = min(self.goal_mask.shape[1] - 1, j + 1)
            k0 = max(0, k - 1); k1 = min(self.goal_mask.shape[2] - 1, k + 1)

            neighborhood = self.goal_mask[i0:i1+1, j0:j1+1, k0:k1+1]
            if neighborhood.any():
                return True

        # Fallback: spherical neighborhood around STN center
        return np.linalg.norm(q.pos - q_goal.pos) <= self.goal_radius


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
            if i % 100 == 0:
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
# Algorithm-1 top-level pieces: LinearPath, CurvePath, MinimumCost
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
    E = Line(q_init, q_goal, step)
    pos = np.vstack([v.pos for v in E])
    if env.collision_free_points(pos):
        c = edge_cost(E, env, L_min, a1, a2)
        log(f"[LinearPath] Straight path is collision-free. Cost={c:.4f}")
        return PathResult(waypoints=E, cost=c)
    log("[LinearPath] Straight path is in collision.")
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
        return PathResult(waypoints=E, cost=c)
    log("[CurvePath] Curved path is in collision or violates curvature constraint.")
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
    Full Algorithm 1:
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
        goal_mask=goal_mask,              # <--- PASS IT IN
    )
    rrt_path = planner.RRTStarPath(q_init, q_goal)
    if rrt_path is not None:
        candidate_paths.append(rrt_path)

    return MinimumCost(candidate_paths)

# ---------------------------------------------------------------------------
# Tiny example to sanity-check
# ---------------------------------------------------------------------------

def _example():
    log("[_example] Running toy example (not used in main optimize_dbs).")
    # Workspace with one spherical obstacle
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
# Generate entry points
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
    z  = env.bounds_max[2]  # top slice

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
# Optimization Function
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
    goal_mask: Optional[np.ndarray] = None,   # <--- NEW PARAM
) -> tuple[Optional[PathResult], list[tuple[Vertex, PathResult]]]:
    """
    Run the planner from each entry on the brain surface and return:
      - best path (minimum cost) or None
      - list of (entry_vertex, PathResult) for all successful entries
    """
    all_results: list[tuple[Vertex, PathResult]] = []

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
            goal_mask=goal_mask,          # <--- PASS IT THROUGH
        )
        if res is not None:
            all_results.append((q_init, res))

    if not all_results:
        return None, []

    best_entry, best_path = min(all_results, key=lambda qr: qr[1].cost)
    return best_path, all_results


# ---------------------------------------------------------------------------
# Plotting Function
# ---------------------------------------------------------------------------

def plot_trajectories_3d_with_structures(
    best: PathResult,
    all_results: list[tuple[Vertex, PathResult]],
    brain_mask: np.ndarray,
    stn_mask: np.ndarray,
    cc_mask: np.ndarray,
    sulci_mask: np.ndarray,
    vent_mask: np.ndarray,
    voxel_size_mm: np.ndarray,
):
    log("[plot_trajectories_3d_with_structures] Building 3D actors for visualization...")
    actors = []

    def to_world(p: np.ndarray) -> np.ndarray:
        # voxel -> mm
        return p * voxel_size_mm

    # --- Structural surfaces ------------------------------------------------
    # Brain outer surface
    log("[plot_trajectories_3d_with_structures] Generating brain surface...")
    brain_vol = Volume(brain_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
    brain_surf = brain_vol.isosurface(0.5).smooth(20)
    brain_surf.c("lightgray").alpha(0.25)
    actors.append(brain_surf)

    # Corpus callosum
    log("[plot_trajectories_3d_with_structures] Generating corpus callosum surface...")
    cc_vol = Volume(cc_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
    cc_surf = cc_vol.isosurface(0.5).smooth(10)
    cc_surf.c("cyan").alpha(0.4)
    actors.append(cc_surf)

    # Sulci (could be big, so you may want decimate)
    log("[plot_trajectories_3d_with_structures] Generating sulci surface...")
    sulci_vol = Volume(sulci_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
    sulci_surf = sulci_vol.isosurface(0.5).decimate(0.8).smooth(10)
    sulci_surf.c("orange").alpha(0.25)
    actors.append(sulci_surf)

    # Ventricles
    log("[plot_trajectories_3d_with_structures] Generating ventricles surface...")
    vent_vol = Volume(vent_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
    vent_surf = vent_vol.isosurface(0.5).smooth(10)
    vent_surf.c("blue").alpha(0.6)
    actors.append(vent_surf)

    # STN (goal region)
    log("[plot_trajectories_3d_with_structures] Generating STN surface...")
    stn_vol = Volume(stn_mask.astype(np.uint8), spacing=voxel_size_mm.tolist())
    stn_surf = stn_vol.isosurface(0.5).smooth(8)
    stn_surf.c("green").alpha(0.9)
    actors.append(stn_surf)

    # --- Trajectories -------------------------------------------------------
    log("[plot_trajectories_3d_with_structures] Adding trajectories...")
    # Non-optimal paths
    for q_init, path in all_results:
        if path is best:
            continue
        pts = np.array([to_world(v.pos) for v in path.waypoints])
        line = Line(pts)
        line.c("lightgray").alpha(0.15).lw(1)
        actors.append(line)

    # Optimal path
    best_pts = np.array([to_world(v.pos) for v in best.waypoints])
    best_line = Line(best_pts)
    best_line.c("red").alpha(1.0).lw(4)
    actors.append(best_line)

    # Entry points (on brain surface)
    entry_pts = [to_world(q.pos) for (q, _) in all_results]
    if entry_pts:
        entries_actor = Points(entry_pts, r=5)
        entries_actor.c("magenta").alpha(0.6)
        actors.append(entries_actor)

    # Goal point (STN center)
    goal_pt = to_world(best.waypoints[-1].pos)
    goal_actor = Points([goal_pt], r=10)
    goal_actor.c("yellow").alpha(1.0)
    actors.append(goal_actor)

    log("[plot_trajectories_3d_with_structures] Rendering with vedo.show()...")
    show(actors, "DBS trajectories on full anatomy", axes=1, viewup="z")


# ---------------------------------------------------------------------------
# Optimization Call
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
    log(f"[optimize_dbs] Voxel size (mm) = {voxel_size}")

    # 1) sample entry points on outer brain surface
    log("[optimize_dbs] Sampling entry points on outer brain surface...")
    entries = sample_surface_entries_on_brain(brain_mask, q_goal, n_entries=60)
    log(f"[optimize_dbs] Generated {len(entries)} surface entries.")

    # 2) optimize over those entries
    r_min_vox = 40  # 1/R = 0.025, R = 40
    log(f"[optimize_dbs] Starting optimization over entries with r_min_vox={r_min_vox}...")
    best, all_results = optimize_over_entries(
        env,
        q_goal,
        entries=entries,
        r_min=r_min_vox,
        step=1.0,
        n_samples=4000,
        goal_radius=2.0,
        a1=0.1,
        a2=0.9,
        w_d=0.7,
        w_theta=0.3,
        goal_mask=stn_mask,      # <--- NEW: goal is the STN volume
    )


    if best is None:
        log("[optimize_dbs] No viable path from any entry. Exiting.")
        print("No viable path from any entry.")
        return None, [], voxel_size

    log(f"[optimize_dbs] Best path cost: {best.cost:.4f}, waypoints={len(best.waypoints)}")
    print(f"Best path cost: {best.cost:.4f}, {len(best.waypoints)} waypoints")

    # 3) directly plot on full anatomy
    log("[optimize_dbs] Plotting trajectories with anatomical structures...")
    plot_trajectories_3d_with_structures(
        best,
        all_results,
        brain_mask,
        stn_mask,
        cc_mask,
        sulci_mask,
        vent_mask,
        voxel_size_mm=voxel_size,
    )

    log("[optimize_dbs] ===== Finished DBS optimization pipeline =====")
    return best, all_results, voxel_size


if __name__ == "__main__":
    optimize_dbs()
