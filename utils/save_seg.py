from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from random import randint
from scipy.optimize import least_squares

# ----------------------------- utilities -------------------------------------
def _fmt(x: float) -> str:
    return f"{float(x):.10g}"

# ------------------------ analytic from quadrics ------------------------------

def _plane_from_quadric(th: np.ndarray,
                            X: np.ndarray = None
                            ) -> Tuple[np.ndarray, np.ndarray, float]:

    G, H, I = map(float, th[6:9])
    nprime = 2.0 * np.array([G, H, I], float)
    n = nprime / (np.linalg.norm(nprime) + 1e-12)

    if X is not None and len(X) > 3:
        d = -float(np.mean(X @ n))
    else:
        d = 0.0  # assume plane through origin if no data
    pos = -d * n
    return n, pos, d

def _sphere_from_quadric(th: np.ndarray, X: np.ndarray = None, r_clip: Tuple[float, float] = (1e-3, 5.0)) -> Tuple[np.ndarray, float]:
    A,B,C,D,E,F,G,H,I,_ = map(float, th)
    Q = np.array([[A,D,E],[D,B,F],[E,F,C]], float)
    b = np.array([G,H,I], float)

    # 1️⃣ center
    try:
        c = -np.linalg.solve(Q, b)
    except np.linalg.LinAlgError:
        c, *_ = np.linalg.lstsq(Q, -b, rcond=None)

    # 2️⃣ isotropic scale
    s = float(np.trace(Q)) / 3.0
    if abs(s) < 1e-15:
        return c, 0.0

    # 3️⃣ radius: if we have points, use median distance; else use algebraic magnitude
    if X is not None and len(X) > 4:
        r = np.mean(np.linalg.norm(X - c, axis=1))
        r = float(np.clip(r, r_clip[0], r_clip[1]))
    else:
        # heuristic: ||b|| ≈ s * ||c||, approximate r ≈ ||c||/√3
        r = float(np.clip(np.linalg.norm(c)/np.sqrt(3.0), r_clip[0], r_clip[1]))

    return c, r

def _cylinder_from_quadric(th: np.ndarray,
                               X: np.ndarray = None,
                               r_clip: Tuple[float, float] = (1e-3, 5.0)
                               ) -> Tuple[np.ndarray, np.ndarray, float]:

    A,B,C,D,E,F,G,H,I,_ = map(float, th)
    Q = np.array([[A, D, E],
                  [D, B, F],
                  [E, F, C]], float)
    b = np.array([G, H, I], float)

    # 1. eigen-decomposition to get axis a and shape scale s
    vals, vecs = np.linalg.eigh(Q)
    idx_axis = np.argmin(np.abs(vals))
    a = vecs[:, idx_axis]
    a /= np.linalg.norm(a) + 1e-12
    s = np.mean(np.delete(vals, idx_axis))
    if s < 0:
        Q, b, s = -Q, -b, -s

    # 2. find a point c0 on the axis (projection)
    P = np.eye(3) - np.outer(a, a)
    M = P @ Q @ P + 1e-8*np.eye(3)
    rhs = -P @ b
    try:
        c0 = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        c0, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    c0 -= a * (a @ c0)

    # 3. estimate radius r (no J)
    if X is not None and len(X) > 3:
        u = X - c0
        lat = u - (u @ a)[:, None] * a[None, :]
        r = np.mean(np.linalg.norm(lat, axis=1))
        r = float(np.clip(r, r_clip[0], r_clip[1]))
    else:
        r = np.linalg.norm(b) / (s + 1e-12)
        r = float(np.clip(r, r_clip[0], r_clip[1]))

    return a, c0, float(r)

def _cone_from_quadric(th: np.ndarray,
                           X: np.ndarray = None
                           ) -> Tuple[np.ndarray, np.ndarray, float]:

    A,B,C,D,E,F,G,H,I,_ = map(float, th)
    Q = np.array([[A,D,E],[D,B,F],[E,F,C]], float)
    b = np.array([G,H,I], float)

    # 1️⃣ Apex (from Qx + b = 0)
    try:
        c0 = -np.linalg.solve(Q, b)
    except np.linalg.LinAlgError:
        c0, *_ = np.linalg.lstsq(Q, -b, rcond=None)

    # 2️⃣ Axis direction from eigen-structure
    Q = 0.5*(Q + Q.T)
    vals, vecs = np.linalg.eigh(Q)
    order = np.argsort(vals)
    vals = vals[order]; vecs = vecs[:, order]

    # detect signature (2+,1-) or (2-,1+)
    if (vals[0]<0 and vals[1]<0 and vals[2]>0):
        idx_axis = 2
        lambda_axis  = vals[2]
        lambda_trans = np.mean(vals[:2])
    elif (vals[0]>0 and vals[1]>0 and vals[2]<0):
        idx_axis = 2
        lambda_axis  = vals[2]
        lambda_trans = np.mean(vals[:2])
    else:
        idx_axis = int(np.argmin(np.abs(vals)))
        lambda_axis  = vals[idx_axis]
        lambda_trans = np.mean(np.delete(vals, idx_axis))

    a = vecs[:, idx_axis]
    a /= np.linalg.norm(a) + 1e-12

    # 3️⃣ Compute opening slope k or half-angle theta
    ratio = max(-lambda_axis / (lambda_trans + 1e-18), 0.0)
    theta = float(np.arctan(np.sqrt(ratio)))  # half-angle in radians

    # 4️⃣ Optional refinement of apex using points (if provided)
    if X is not None and len(X) > 6:
        # Fit apex along axis minimizing cone residual
        t = (X - c0) @ a
        lat = X - c0 - t[:,None]*a[None,:]
        k_est = np.mean(np.linalg.norm(lat, axis=1) / (np.abs(t)+1e-9))
        k_est = np.clip(k_est, 1e-3, 10.0)
        theta = np.arctan(k_est)
        d_shift = np.mean(t - np.linalg.norm(lat, axis=1)/k_est)
        c0 = c0 + d_shift * a

    return a, c0, theta


# --------------------------- points-only fits --------------------------------
def _plane_from_points(Pg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    if Pg.shape[0] < 3:
        return np.array([0,0,1.0], float), np.zeros(3,float), 0.0
    c = Pg.mean(0); X = Pg - c
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1]; n = n/(np.linalg.norm(n)+1e-12)
    d = float(-c.dot(n))
    return n, c, d

def _sphere_from_points(Pg: np.ndarray) -> Tuple[np.ndarray, float]:
    if Pg.shape[0] < 3:
        return np.zeros(3, float), 0.0
    A_ = np.hstack([2*Pg, np.ones((Pg.shape[0],1))])
    b_ = (Pg**2).sum(axis=1, keepdims=True)
    sol, *_ = np.linalg.lstsq(A_, b_, rcond=None)
    c  = sol[:3,0]
    d  = sol[3,0]
    r  = float(np.sqrt(max(d + float(c.dot(c)), 0.0)))
    return c, r

def _circle_fit_on_plane(Pg: np.ndarray, a: np.ndarray):
    """Given axis a, fit circle in plane ⟂ a. Return (radius, point_on_axis, residual)."""
    a = a / (np.linalg.norm(a)+1e-12)
    tmp = np.array([1.0,0.0,0.0])
    if abs(np.dot(tmp, a)) > 0.9: tmp = np.array([0.0,1.0,0.0])
    u = np.cross(a, tmp); u /= (np.linalg.norm(u)+1e-12)
    v = np.cross(a, u)

    t = Pg @ a
    q = Pg - np.outer(t, a)
    x = q @ u; y = q @ v

    A = np.column_stack((2*x, 2*y, np.ones_like(x)))
    b = x*x + y*y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, _ = sol
    except Exception:
        cx, cy = np.median(x), np.median(y)

    c_perp = cx*u + cy*v
    t0 = np.median(t)
    x0 = c_perp + t0*a
    r  = float(np.median(np.sqrt((x - cx)**2 + (y - cy)**2)))
    res = float(np.median(np.abs(np.sqrt((x - cx)**2 + (y - cy)**2) - r)))
    return r, x0, res

def cylinder_residual(params, points, lambda_center=1e-2):
    """Residual: distance to cylinder surface + soft penalty for far center."""
    c = params[:3]
    a = params[3:6]; a = a / (np.linalg.norm(a) + 1e-12)
    r = params[6]

    v = points - c
    d_axis = np.dot(v, a)[:, None] * a
    dist = np.linalg.norm(v - d_axis, axis=1) - r
    reg = lambda_center * c
    return np.concatenate([dist, reg])


def _cylinder_from_points(Pg: np.ndarray, refine=True, max_iter=100, verbose=0):
    """Hybrid cylinder fit: geometric init + optional least-squares refinement."""
    if Pg.shape[0] < 3:
        return np.array([0,0,1.0], float), np.zeros(3,float), 0.0

    # --- 1️⃣ PCA to get candidate axes ---
    c = Pg.mean(0)
    X = Pg - c
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    candidates = [vh[0], vh[1], vh[2]]

    # --- 2️⃣ try all candidate axes, pick best circle fit ---
    best = (None, None, np.inf)
    for a in candidates:
        a = a / (np.linalg.norm(a)+1e-12)
        r, x0, res = _circle_fit_on_plane(Pg, a)
        if res < best[2]:
            best = (a, (x0, r), res)
    a0, (x0, r0), _ = best

    if not refine:
        return a0, x0, r0

    # --- 3️⃣ least-squares refinement ---
    params0 = np.hstack([x0, a0, r0])
    try:
        res = least_squares(
            cylinder_residual, params0,
            args=(Pg, 1e-2),
            method='trf', max_nfev=max_iter,
            bounds=([-5,-5,-5,-np.inf,-np.inf,-np.inf,0],
                    [ 5, 5, 5, np.inf, np.inf, np.inf, np.inf]),
            verbose=verbose
        )
        c = res.x[:3]
        a = res.x[3:6] / np.linalg.norm(res.x[3:6])
        r = res.x[6]
    except Exception:
        c, a, r = x0, a0, r0

    return a, c, float(r)

def cone_residual(params, P, lambda_center=1e-3):
    """Residual for least-squares cone fitting with apex regularization."""
    v = params[:3]                     # Apex
    a = params[3:6]
    a /= np.linalg.norm(a) + 1e-12     # Normalize axis
    theta = np.clip(params[6], 1e-6, np.pi/2 - 1e-6)
    tan_theta = np.tan(theta)

    V = P - v
    h = V @ a
    r = np.linalg.norm(V - np.outer(h, a), axis=1)
    valid = np.abs(h) > 1e-8
    residuals = np.zeros_like(h)
    residuals[valid] = r[valid] / np.abs(h[valid]) - tan_theta

    # regularization: keep apex near origin
    reg = lambda_center * v
    return np.concatenate([residuals, reg])

def _cone_from_points(Pg: np.ndarray, refine=True, max_iter=100, lambda_apex=1e-3, verbose=0) -> Tuple[np.ndarray, np.ndarray, float]:
    """(axis_dir, apex, half_angle) from points only."""
    if Pg.shape[0] < 5:
        return np.array([0,0,1.0], float), Pg.mean(0) if Pg.size else np.zeros(3,float), 0.0
    c = Pg.mean(0); X = Pg - c
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    # try both directions around smallest and largest spread
    axes = [vh[0], vh[-1]]
    best = (None, None, np.inf)
    for a in axes:
        a = a/(np.linalg.norm(a)+1e-12)
        t = X @ a
        r = np.linalg.norm(X - np.outer(t, a), axis=1)
        A = np.column_stack([t, np.ones_like(t)])
        k_b, *_ = np.linalg.lstsq(A, r, rcond=None)
        k, b = k_b
        s0 = -b/k if abs(k) > 1e-9 else 0.0
        apex = c + s0*a
        theta = float(np.arctan(abs(k)))
        # residual: MAD of |r - (k*t + b)|
        res = float(np.median(np.abs(r - (k*t + b))))
        if res < best[2]:
            best = (a, (apex, theta), res)
    a = best[0]; apex, theta = best[1]
    if np.median((Pg - apex) @ a) < 0: a = -a
    
    degenerate = (theta < np.deg2rad(5.0)) or (theta > np.deg2rad(85.0))
    if not refine or degenerate:
        return a, apex, theta
    
    
    # --- least-squares refinement ---
    try:
        params0 = np.concatenate([apex, a, [theta]])
        res = least_squares(
            cone_residual, params0, args=(Pg, lambda_apex),
            method='trf', max_nfev=max_iter,
            verbose=verbose
        )

        apex = res.x[:3]
        a = res.x[3:6] / np.linalg.norm(res.x[3:6])
        theta = np.clip(res.x[6], 0, np.pi/2)
        # check denegeracy after refinement
        if (theta < np.deg2rad(5.0)) or (theta > np.deg2rad(85.0)):
            apex, a, theta = params0[0:3], params0[3:6]/np.linalg.norm(params0[3:6]), params0[6]
            return a, apex, theta
    except Exception:
        apex, a, theta = params0[0:3], params0[3:6]/np.linalg.norm(params0[3:6]), params0[6]
    
    return a, apex, theta

# ------------------------- main writer ---------------------------------
def save_seg(points: np.ndarray,
             filepath: str,
             group_info: Dict[int, Any] | List[Dict[str, Any]] | None,
             group_col: int = 3,
             normals: Optional[np.ndarray] = None,
             *,
             param_mode: str = "analytic",   # "analytic" | "points"
             swap_cyl_sph: bool = True       # swap 1<->2 on output (your working setup)
             ) -> None:
    """
    Write SEG (ASCII PLY).

    param_mode:
      - "analytic": derive all parameters strictly from quadrics (if provided).
      - "points"  : derive all parameters strictly from points (ignore quadrics).
    """
    pts = np.asarray(points)
    xyz = np.asarray(pts[:, :3], dtype=float)
    gids_raw = np.asarray(pts[:, group_col], dtype=int)
    N = xyz.shape[0]

    if normals is None or np.asarray(normals).shape != (N, 3):
        normals = np.zeros((N, 3), dtype=float)

    uniq_vals, first_idx = np.unique(gids_raw, return_index=True)
    order = np.argsort(first_idx)
    uniq_gids = uniq_vals[order]
    remap = {int(g): i for i, g in enumerate(uniq_gids)}
    segments = np.vectorize(remap.get)(gids_raw).astype(int)

    # build lookup
    lookup = group_info if isinstance(group_info, dict) else {int(e["id"]): e for e in (group_info or [])}

    per_seg_points = [xyz[gids_raw == gid] for gid in uniq_gids]
    shapes: List[Tuple[np.ndarray,np.ndarray,Tuple[int,int,int],float,float,int]] = []

    for sid, gid in enumerate(uniq_gids):
        e = lookup.get(int(gid), {})
        seg_type = int(e.get("type", 0))  # 0 plane, 1 sphere, 2 cylinder, 3 cone

        
        if swap_cyl_sph:
            seg_type = {0: 0, 1: 2, 2: 1, 3: 3}.get(seg_type, seg_type)

        col = e.get("color", (randint(96, 224), randint(96, 224), randint(96, 224)))
        r,g,b = [int(max(0, min(255, v))) for v in col]

        th10 = np.asarray(e.get("parameters", []), dtype=float).reshape(-1)[:10]
        Pg = per_seg_points[sid]

        pos = np.zeros(3, float)
        direc = np.array([0.0, 0.0, 1.0], float)
        r1 = 0.0; r2 = 0.0

        if seg_type == 0:  # plane
            if param_mode == "analytic" and th10.size == 10:
                n, pos, d = _plane_from_quadric(th10)
            else:
                n, pos, d = _plane_from_points(Pg)
            direc = n; r1 = d; r2 = 0.0

        elif seg_type == 1:  # sphere
            if param_mode == "analytic" and th10.size == 10:
                pos, r1 = _sphere_from_quadric(th10)
            else:
                pos, r1 = _sphere_from_points(Pg)
            direc = np.array([0.0,0.0,1.0], float); r2 = 0.0

        elif seg_type == 2:  # cylinder
            if param_mode == "analytic" and th10.size == 10:
                direc, pos, r1 = _cylinder_from_quadric(th10)
            else:
                direc, pos, r1 = _cylinder_from_points(Pg)
            r2 = 0.0

        elif seg_type == 3:  # cone
            if param_mode == "analytic" and th10.size == 10:
                direc, pos, r1 = _cone_from_quadric(th10)
            else:
                direc, pos, r1 = _cone_from_points(Pg)

        else:
            raise ValueError(f"Unsupported segment type code: {seg_type}")

        shapes.append((pos.astype(float), direc.astype(float), (r,g,b), float(r1), float(r2), int(seg_type)))

    # ------------------------------ write file --------------------------------
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Saved by Easy3D\n")
        f.write("comment segment\n")
        f.write(f"element vertex {N}\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property int segment\n")
        f.write(f"element shapes {len(shapes)}\n")
        f.write("property float pos_x\n")
        f.write("property float pos_y\n")
        f.write("property float pos_z\n")
        f.write("property float dir_x\n")
        f.write("property float dir_y\n")
        f.write("property float dir_z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float r1\n")
        f.write("property float r2\n")
        f.write("property int type\n")
        f.write("end_header\n")

        for i in range(N):
            nx, ny, nz = normals[i]
            x,  y,  z  = xyz[i]
            f.write(f"{_fmt(nx)} {_fmt(ny)} {_fmt(nz)} {_fmt(x)} {_fmt(y)} {_fmt(z)} {int(segments[i])}\n")

        for pos, direc, (r,g,b), r1, r2, t in shapes:
            f.write(
                f"{_fmt(pos[0])} {_fmt(pos[1])} {_fmt(pos[2])} "
                f"{_fmt(direc[0])} {_fmt(direc[1])} {_fmt(direc[2])} "
                f"{r} {g} {b} {_fmt(r1)} {_fmt(r2)} {t}\n"
            )
