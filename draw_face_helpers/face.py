import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def _ellipse_path(cx, cy, width, height, n=240):
    t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    x = cx + 0.5 * float(width) * np.cos(t)
    y = cy + 0.5 * float(height) * np.sin(t)

    verts = [(float(x[0]), float(y[0]))]
    codes = [Path.MOVETO]
    for xi, yi in zip(x[1:], y[1:]):
        verts.append((float(xi), float(yi)))
        codes.append(Path.LINETO)
    verts.append((0.0, 0.0))
    codes.append(Path.CLOSEPOLY)
    return Path(verts, codes)


# ----------------------------
# Face patch: "rect" (your current rounded skewed rectangle) OR "oval" (ellipse)
# ----------------------------
def draw_face_patch(width, height, angle_top_to_bottom=0,
                    rounding_top=0.0, rounding_bottom=0.0,
                    cx=0.5, cy=0.5,
                    facecolor="none", edgecolor="black", lw=10,
                    zorder=20,
                    shape="rect",          # NEW: "rect" | "oval"
                    oval_n=240):            # NEW: ellipse smoothness

    shape = str(shape).lower().strip()
    if shape not in ("rect", "oval", "ellipse"):
        raise ValueError("shape must be 'rect' or 'oval' (aka 'ellipse')")

    # ---- OVAL MODE
    if shape in ("oval", "ellipse"):
        path = _ellipse_path(cx, cy, width, height, n=oval_n)
        return PathPatch(
            path,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
            joinstyle="round",
            capstyle="round",
            zorder=zorder,
        )

    # ---- RECT MODE (your current implementation)
    theta = np.deg2rad(angle_top_to_bottom)
    dx = height * np.tan(theta)

    w_top = width
    w_bot = width + 2 * dx

    y_top = cy + height / 2
    y_bot = cy - height / 2

    tl = np.array([cx - w_top / 2, y_top])
    tr = np.array([cx + w_top / 2, y_top])
    br = np.array([cx + w_bot / 2, y_bot])
    bl = np.array([cx - w_bot / 2, y_bot])

    corners = [tl, tr, br, bl]  # clockwise
    r_corner = np.array([rounding_top, rounding_top, rounding_bottom, rounding_bottom], dtype=float)

    edge_len = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    for i in range(4):
        adj_min = min(edge_len[i - 1], edge_len[i])
        r_corner[i] = max(0.0, min(float(r_corner[i]), 0.49 * float(adj_min)))

    p_in, p_out = [], []
    for i in range(4):
        p = corners[i]
        p_prev = corners[i - 1]
        p_next = corners[(i + 1) % 4]
        r = float(r_corner[i])

        u_prev = (p_prev - p); u_prev /= np.linalg.norm(u_prev)
        u_next = (p_next - p); u_next /= np.linalg.norm(u_next)

        p_in.append(p + u_prev * r)
        p_out.append(p + u_next * r)

    verts, codes = [], []
    verts.append(tuple(p_out[0])); codes.append(Path.MOVETO)

    for i in [1, 2, 3]:
        verts.append(tuple(p_in[i]));     codes.append(Path.LINETO)
        verts.append(tuple(corners[i]));  codes.append(Path.CURVE3)
        verts.append(tuple(p_out[i]));    codes.append(Path.CURVE3)

    verts.append(tuple(p_in[0]));     codes.append(Path.LINETO)
    verts.append(tuple(corners[0]));  codes.append(Path.CURVE3)
    verts.append(tuple(p_out[0]));    codes.append(Path.CURVE3)

    verts.append((0.0, 0.0)); codes.append(Path.CLOSEPOLY)

    return PathPatch(
        Path(verts, codes),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        joinstyle="round",
        capstyle="round",
        zorder=zorder
    )


# ----------------------------
# Get dense face boundary polygon in data coords
# ----------------------------
def _face_boundary_polygon(face_patch, interp=120):
    path_data = face_patch.get_path().transformed(face_patch.get_patch_transform())
    path_dense = path_data.interpolated(interp)
    polys = path_dense.to_polygons(closed_only=True)
    if not polys:
        raise ValueError("Could not extract polygon from face patch.")
    return max(polys, key=lambda a: a.shape[0])
