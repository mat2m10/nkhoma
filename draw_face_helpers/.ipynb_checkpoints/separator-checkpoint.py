import numpy as np
from .geometry import _px_to_data_dy
from .face import _face_boundary_polygon


def build_separator(ax, face_patch, cx=0.5, cy=0.5,
                    y_pos_px=15, curvature_px=35,
                    n=600, boundary_interp=200,
                    curve_dir="up"):   # NEW: "down" (smile) or "up" (sad)
    ax.figure.canvas.draw()

    dy = _px_to_data_dy(ax, y_pos_px)
    dcurve = abs(_px_to_data_dy(ax, curvature_px))
    y_sep = cy + dy

    curve_dir = str(curve_dir).lower().strip()
    if curve_dir in ("down", "smile", "convex", "u"):
        sign = -1.0
    elif curve_dir in ("up", "sad", "concave", "n"):
        sign = +1.0
    else:
        raise ValueError("curve_dir must be 'down' or 'up' (aliases: smile/sad)")

    poly = _face_boundary_polygon(face_patch, interp=boundary_interp)

    inter = []
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(poly[:-1], poly[1:])):
        if (y1 <= y_sep <= y2) or (y2 <= y_sep <= y1):
            dyseg = (y2 - y1)
            if abs(dyseg) < 1e-12:
                continue
            tseg = (y_sep - y1) / dyseg
            if 0.0 <= tseg <= 1.0:
                x = x1 + tseg * (x2 - x1)
                inter.append({"x": float(x), "y": float(y_sep), "i": i, "t": float(tseg)})

    if len(inter) < 2:
        y_min = float(np.min(poly[:, 1])); y_max = float(np.max(poly[:, 1]))
        raise ValueError(
            f"Separator y={y_sep:.4f} did not intersect face. "
            f"Face y-range=[{y_min:.4f}, {y_max:.4f}]."
        )

    inter = sorted(inter, key=lambda d: d["x"])
    inter2 = [inter[0]]
    for d in inter[1:]:
        if abs(d["x"] - inter2[-1]["x"]) > 1e-6:
            inter2.append(d)
    if len(inter2) < 2:
        raise ValueError("Intersections collapsed after dedupe.")

    L, R = inter2[0], inter2[-1]

    p0 = np.array([L["x"], y_sep], dtype=float)
    p2 = np.array([R["x"], y_sep], dtype=float)
    pm = 0.5 * (p0 + p2)

    # ctrl y determines curvature orientation
    ctrl = np.array([pm[0], pm[1] + sign * dcurve], dtype=float)

    t = np.linspace(0.0, 1.0, int(n))
    sep_pts = ((1 - t)[:, None] ** 2) * p0 + (2 * (1 - t)[:, None] * t[:, None]) * ctrl + (t[:, None] ** 2) * p2
    return y_sep, sep_pts, L, R, poly
