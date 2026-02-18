
import numpy as np
import matplotlib.lines as mlines
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from .separator import build_separator
from .geometry import _px_to_data_dy

# ----------------------------
# Draw fills + separator line
# ----------------------------
def draw_face_brain(ax, face_patch,
                    cx=0.5, cy=0.5,
                    y_pos_px=15,
                    curvature_px=35,
                    curve_dir="down",
                    sep_n=600,
                    boundary_interp=200,
                    fill_pad_px=0,
                    lw=10,
                    brain_color="#cfe8ff",
                    face_color="#ffe0cc",
                    line_color="black",
                    z_fill=10,
                    z_line=30):

    y_sep, sep_pts, L, R, poly = build_separator(
        ax, face_patch, cx=cx, cy=cy,
        y_pos_px=y_pos_px, curvature_px=curvature_px,
        n=sep_n, boundary_interp=boundary_interp,curve_dir=curve_dir,
    )

    pad_dy = _px_to_data_dy(ax, fill_pad_px) if fill_pad_px else 0.0
    if pad_dy != 0.0:
        sep_top = sep_pts.copy(); sep_top[:, 1] -= abs(pad_dy)
        sep_bot = sep_pts.copy(); sep_bot[:, 1] += abs(pad_dy)
    else:
        sep_top = sep_pts
        sep_bot = sep_pts

    def arc_points(A, B):
        iA, iB = A["i"], B["i"]
        pts = [np.array([A["x"], A["y"]], dtype=float)]
        j = iA + 1
        N = len(poly) - 1
        while True:
            pts.append(poly[j % N].astype(float))
            if (j % N) == (iB % N):
                break
            j += 1
        pts[-1] = np.array([B["x"], B["y"]], dtype=float)
        return np.vstack(pts)

    arc_LR = arc_points(L, R)
    arc_RL = arc_points(R, L)

    if np.mean(arc_LR[:, 1]) >= np.mean(arc_RL[:, 1]):
        top_arc, bot_arc = arc_LR, arc_RL
    else:
        top_arc, bot_arc = arc_RL, arc_LR

    top_poly = np.vstack([top_arc, sep_top[::-1, :]])
    bot_poly = np.vstack([sep_bot, bot_arc[::-1, :]])

    def poly_to_patch(pts, color, z):
        verts = [tuple(pts[0])]
        codes = [Path.MOVETO]
        for p in pts[1:]:
            verts.append(tuple(p))
            codes.append(Path.LINETO)
        verts.append((0.0, 0.0)); codes.append(Path.CLOSEPOLY)
        patch = PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=z)
        patch.set_clip_path(face_patch)
        ax.add_patch(patch)
        return patch

    face_fill = poly_to_patch(bot_poly, face_color, z_fill)

    # NOTE: keep return order the same, but internally name this face_outline (per your request)
    face_outline = poly_to_patch(top_poly, brain_color, z_fill + 1)

    line = mlines.Line2D(sep_pts[:, 0], sep_pts[:, 1],
                         lw=lw, color=line_color,
                         solid_capstyle="round",
                         zorder=z_line)
    line.set_clip_path(face_patch)
    ax.add_line(line)

    # return stays the same: (brain_fill, face_fill, line)
    return face_outline, face_fill, line