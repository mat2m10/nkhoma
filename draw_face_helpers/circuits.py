import numpy as np
import matplotlib.lines as mlines
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path

from .geometry import _px_to_data_dx, _px_to_data_dy
from .separator import build_separator


def add_one_brain_circuit_from_brain_top(ax, face_patch,
                                        center_shift=0.0,   # fraction of brain width
                                        length_px=78,       # pixels downward from the top edge
                                        node_r=0.020,
                                        node_facecolor="white",
                                        node_edgecolor="black",
                                        wire_color="black",
                                        wire_lw=7,          # points
                                        zorder=26,
                                        cap_safety=1.10):
    """
    Same behavior: takes whatever you pass (now face_outline) and internally treats it
    as the brain region clip (previously 'brain_fill').
    """

    def top_y_at_x_from_poly(poly_xy, xq, eps=1e-9):
        if not np.allclose(poly_xy[0], poly_xy[-1]):
            poly_xy = np.vstack([poly_xy, poly_xy[0]])

        ys = []
        for (x1, y1), (x2, y2) in zip(poly_xy[:-1], poly_xy[1:]):
            dx = x2 - x1

            if abs(dx) <= eps:
                if abs(xq - x1) <= eps:
                    ys.append(float(y1))
                    ys.append(float(y2))
                continue

            if (min(x1, x2) - eps) <= xq <= (max(x1, x2) + eps):
                t = (xq - x1) / dx
                if (-eps) <= t <= (1.0 + eps):
                    ys.append(float(y1 + t * (y2 - y1)))

        if not ys:
            return None
        return max(ys)

    face_outline = face_patch

    brain_path = face_outline.get_path().transformed(face_outline.get_patch_transform())
    polys = brain_path.to_polygons(closed_only=True)
    if not polys:
        return None, None
    poly = max(polys, key=len)

    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    W = float(xmax - xmin)

    x_center = 0.5 * (xmin + xmax)
    x = float(x_center + float(center_shift) * W)

    y_top = top_y_at_x_from_poly(poly, x)
    if y_top is None:
        return None, None

    cap_radius_px = (float(wire_lw) * ax.figure.dpi / 72.0) * 0.5
    cap_dy = abs(_px_to_data_dy(ax, cap_radius_px * float(cap_safety)))
    y_start = float(y_top + cap_dy)

    dy = abs(_px_to_data_dy(ax, float(length_px)))
    y_end = float(y_top - dy)

    wire = mlines.Line2D([x, x], [y_start, y_end],
                         lw=wire_lw, color=wire_color,
                         solid_capstyle="round",
                         zorder=zorder)
    wire.set_clip_path(face_outline)
    ax.add_line(wire)

    node = Circle((x, y_end), node_r,
                  facecolor=node_facecolor,
                  edgecolor=node_edgecolor,
                  linewidth=float(wire_lw) * 0.55,
                  zorder=zorder + 1)
    node.set_clip_path(face_outline)
    ax.add_patch(node)

    return node, wire


def add_bent_brain_circuit_from_face_outline(ax, face_patch,
                                            cx=0.5, cy=0.5,
                                            y_pos_px=15,
                                            curvature_px=35,
                                            boundary_interp=220,
                                            sep_n=700,
                                            fill_pad_px=2,
                                            center_shift=0.0,
                                            length_px=90,
                                            bend_px=28,
                                            bend_at=0.45,
                                            node_r=0.020,
                                            node_facecolor="white",
                                            node_edgecolor="black",
                                            wire_color="black",
                                            wire_lw=7,
                                            zorder=26,
                                            cap_safety=1.10):
    """
    Bent brain circuit that RECEIVES face_patch (face_outline) and clips wire+node to the computed brain region.
    """

    y_sep, sep_pts, L, R, poly = build_separator(
        ax, face_patch, cx=cx, cy=cy,
        y_pos_px=y_pos_px, curvature_px=curvature_px,
        n=sep_n, boundary_interp=boundary_interp
    )

    pad_dy = _px_to_data_dy(ax, fill_pad_px) if fill_pad_px else 0.0
    if pad_dy != 0.0:
        sep_top = sep_pts.copy()
        sep_top[:, 1] -= abs(pad_dy)
    else:
        sep_top = sep_pts

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

    top_arc = arc_LR if (np.mean(arc_LR[:, 1]) >= np.mean(arc_RL[:, 1])) else arc_RL
    top_poly = np.vstack([top_arc, sep_top[::-1, :]])

    verts = [tuple(top_poly[0])]
    codes = [Path.MOVETO]
    for p in top_poly[1:]:
        verts.append(tuple(p))
        codes.append(Path.LINETO)
    verts.append((0.0, 0.0))
    codes.append(Path.CLOSEPOLY)

    brain_clip = PathPatch(Path(verts, codes), facecolor="none", edgecolor="none")
    brain_clip.set_clip_path(face_patch)
    brain_clip.set_alpha(0.0)
    brain_clip.set_zorder(-999)
    ax.add_patch(brain_clip)

    xmin, ymin = top_poly.min(axis=0)
    xmax, ymax = top_poly.max(axis=0)
    W = float(xmax - xmin)
    x_center = 0.5 * (xmin + xmax)
    x0 = float(x_center + float(center_shift) * W)

    def top_y_at_x(polyline_xy, xq, eps=1e-9):
        ys = []
        for (x1, y1), (x2, y2) in zip(polyline_xy[:-1], polyline_xy[1:]):
            dx = x2 - x1
            if abs(dx) <= eps:
                if abs(xq - x1) <= eps:
                    ys.append(float(y1)); ys.append(float(y2))
                continue
            if (min(x1, x2) - eps) <= xq <= (max(x1, x2) + eps):
                t = (xq - x1) / dx
                if (-eps) <= t <= (1.0 + eps):
                    ys.append(float(y1 + t * (y2 - y1)))
        return max(ys) if ys else None

    y_top = top_y_at_x(top_arc, x0)
    if y_top is None:
        return None, None

    cap_radius_px = (float(wire_lw) * ax.figure.dpi / 72.0) * 0.5
    cap_dy = abs(_px_to_data_dy(ax, cap_radius_px * float(cap_safety)))
    y0 = float(y_top + cap_dy)

    dy_total = abs(_px_to_data_dy(ax, float(length_px)))
    dx_bend = _px_to_data_dx(ax, float(bend_px))

    y1 = float(y_top - dy_total * float(bend_at))
    x1 = float(x0 + dx_bend)

    y2 = float(y_top - dy_total)
    x2 = float(x1)

    wire = mlines.Line2D([x0, x1, x2], [y0, y1, y2],
                         lw=wire_lw, color=wire_color,
                         solid_capstyle="round",
                         solid_joinstyle="round",
                         zorder=zorder)
    wire.set_clip_path(brain_clip)
    ax.add_line(wire)

    node = Circle((x2, y2), node_r,
                  facecolor=node_facecolor,
                  edgecolor=node_edgecolor,
                  linewidth=float(wire_lw) * 0.55,
                  zorder=zorder + 1)
    node.set_clip_path(brain_clip)
    ax.add_patch(node)

    return node, wire


def add_s_double_node_circuit_from_brain_top(ax, face_patch,
                                             center_shift=0.0,
                                             length_px=95,
                                             amp_px=18,
                                             waves=1.0,
                                             n=220,
                                             node_t1=0.45,
                                             node_t2=0.85,
                                             node_r=0.020,
                                             node_facecolor="white",
                                             node_edgecolor="black",
                                             wire_color="black",
                                             wire_lw=7,
                                             zorder=26,
                                             cap_safety=1.10):
    """
    Same behavior: takes whatever you pass (now face_outline) and internally treats it
    as the brain region clip (previously 'brain_fill').
    """

    def top_y_at_x_from_poly(poly_xy, xq, eps=1e-9):
        if not np.allclose(poly_xy[0], poly_xy[-1]):
            poly_xy = np.vstack([poly_xy, poly_xy[0]])

        ys = []
        for (x1, y1), (x2, y2) in zip(poly_xy[:-1], poly_xy[1:]):
            dx = x2 - x1

            if abs(dx) <= eps:
                if abs(xq - x1) <= eps:
                    ys.append(float(y1))
                    ys.append(float(y2))
                continue

            if (min(x1, x2) - eps) <= xq <= (max(x1, x2) + eps):
                t = (xq - x1) / dx
                if (-eps) <= t <= (1.0 + eps):
                    ys.append(float(y1 + t * (y2 - y1)))

        return max(ys) if ys else None

    face_outline = face_patch

    brain_path = face_outline.get_path().transformed(face_outline.get_patch_transform())
    polys = brain_path.to_polygons(closed_only=True)
    if not polys:
        return None, None, None

    poly = max(polys, key=len)
    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    W = float(xmax - xmin)

    x_center = 0.5 * (xmin + xmax)
    x0 = float(x_center + float(center_shift) * W)

    y_top = top_y_at_x_from_poly(poly, x0)
    if y_top is None:
        return None, None, None

    cap_radius_px = (float(wire_lw) * ax.figure.dpi / 72.0) * 0.5
    cap_dy = abs(_px_to_data_dy(ax, cap_radius_px * float(cap_safety)))
    y_start = float(y_top + cap_dy)

    dy_total = abs(_px_to_data_dy(ax, float(length_px)))
    dx_amp = _px_to_data_dx(ax, float(amp_px))

    t = np.linspace(0.0, 1.0, int(n))
    xs = x0 + dx_amp * np.sin(2.0 * np.pi * float(waves) * t)
    ys = y_start - dy_total * t

    wire = mlines.Line2D(xs, ys,
                         lw=wire_lw, color=wire_color,
                         solid_capstyle="round",
                         solid_joinstyle="round",
                         zorder=zorder)
    wire.set_clip_path(face_outline)
    ax.add_line(wire)

    def add_node_at(tt):
        tt = float(np.clip(tt, 0.0, 1.0))
        i = int(round(tt * (len(t) - 1)))
        node = Circle((float(xs[i]), float(ys[i])), node_r,
                      facecolor=node_facecolor,
                      edgecolor=node_edgecolor,
                      linewidth=float(wire_lw) * 0.55,
                      zorder=zorder + 1)
        node.set_clip_path(face_outline)
        ax.add_patch(node)
        return node

    node1 = add_node_at(node_t1)
    node2 = add_node_at(node_t2)

    return node1, node2, wire


def add_rect_bent_double_node_from_brain_top(ax, face_patch,
                                            center_shift=0.0,
                                            length_px=80,
                                            bend_px=22,
                                            bend_at=0.35,
                                            bend_dir=+1,
                                            node1_on="vertical1",   # NEW: "vertical1" | "horizontal" | "vertical2"
                                            node1_u=0.6,            # NEW: 0..1 along that segment
                                            node_facecolor1="#4aa3c7",
                                            node_facecolor2="#4aa3c7",
                                            node_edgecolor="black",
                                            node_r=0.020,
                                            wire_color="black",
                                            wire_lw=7,
                                            zorder=26,
                                            cap_safety=1.10,
                                            corner_px=6,
                                            corner_n=20):
    """
    Rect bent wire with two nodes:
    - node2 is always at the end of the wire
    - bend_dir mirrors the bend (+1 right, -1 left)
    - node1 placement is controllable by segment (node1_on) + fraction (node1_u)
    """

    def top_y_at_x_from_poly(poly_xy, xq, eps=1e-9):
        if not np.allclose(poly_xy[0], poly_xy[-1]):
            poly_xy = np.vstack([poly_xy, poly_xy[0]])
        ys = []
        for (x1, y1), (x2, y2) in zip(poly_xy[:-1], poly_xy[1:]):
            dx = x2 - x1
            if abs(dx) <= eps:
                if abs(xq - x1) <= eps:
                    ys.append(float(y1)); ys.append(float(y2))
                continue
            if (min(x1, x2) - eps) <= xq <= (max(x1, x2) + eps):
                t = (xq - x1) / dx
                if (-eps) <= t <= (1.0 + eps):
                    ys.append(float(y1 + t * (y2 - y1)))
        return max(ys) if ys else None

    face_outline = face_patch

    brain_path = face_outline.get_path().transformed(face_outline.get_patch_transform())
    polys = brain_path.to_polygons(closed_only=True)
    if not polys:
        return None, None, None

    poly = max(polys, key=len)
    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    W = float(xmax - xmin)

    x_center = 0.5 * (xmin + xmax)
    x0 = float(x_center + float(center_shift) * W)

    y_top = top_y_at_x_from_poly(poly, x0)
    if y_top is None:
        return None, None, None

    cap_radius_px = (float(wire_lw) * ax.figure.dpi / 72.0) * 0.5
    cap_dy = abs(_px_to_data_dy(ax, cap_radius_px * float(cap_safety)))
    y0 = float(y_top + cap_dy)

    dy_total = abs(_px_to_data_dy(ax, float(length_px)))

    dx_step = _px_to_data_dx(ax, float(bend_px))
    s = 1.0 if float(bend_dir) >= 0 else -1.0   # +right, -left
    dx_step *= s

    y_corner = float(y_top - dy_total * float(bend_at))
    x0_base = float(x0)
    x_right = float(x0 + dx_step)
    y_end   = float(y_top - dy_total)

    r = max(0.0, float(_px_to_data_dx(ax, float(corner_px))))

    max_r1 = abs(y_corner - y0) * 0.45
    max_r2 = abs(x_right - x0_base) * 0.45
    max_r3 = abs(y_end - y_corner) * 0.45
    r = min(r, max_r1, max_r2, max_r3)

    # --- node1 position (stable under mirroring)
    node1_on_ = str(node1_on).lower().strip()
    u = float(np.clip(node1_u, 0.0, 1.0))

    if node1_on_ == "vertical1":
        p1 = (float(x0_base), float(y0 + (y_corner - y0) * u))
    elif node1_on_ == "horizontal":
        p1 = (float(x0_base + (x_right - x0_base) * u), float(y_corner))
    elif node1_on_ == "vertical2":
        p1 = (float(x_right), float(y_corner + (y_end - y_corner) * u))
    else:
        raise ValueError("node1_on must be 'vertical1', 'horizontal', or 'vertical2'")

    # --- build rounded polyline
    xA, yA = float(x0_base), float(y0)
    xB, yB = float(x0_base), float(y_corner + r)

    c1x, c1y = float(x0_base + s * r), float(y_corner + r)
    xD, yD   = float(x_right - s * r), float(y_corner)
    c2x, c2y = float(x_right - s * r), float(y_corner - r)

    xF, yF = float(x_right), float(y_end)

    if r > 0:
        if s > 0:
            ang1 = np.linspace(np.pi, 1.5*np.pi, int(corner_n))
            ang2 = np.linspace(0.5*np.pi, 0.0, int(corner_n))
        else:
            ang1 = np.linspace(0.0, -0.5*np.pi, int(corner_n))
            ang2 = np.linspace(0.5*np.pi, np.pi, int(corner_n))

        arc1x = c1x + r * np.cos(ang1)
        arc1y = c1y + r * np.sin(ang1)

        arc2x = c2x + r * np.cos(ang2)
        arc2y = c2y + r * np.sin(ang2)
    else:
        arc1x = np.array([x0_base], dtype=float); arc1y = np.array([y_corner], dtype=float)
        arc2x = np.array([x_right], dtype=float); arc2y = np.array([y_corner], dtype=float)

    xs = np.concatenate([[xA, xB], arc1x, [xD], arc2x, [xF]]).astype(float)
    ys = np.concatenate([[yA, yB], arc1y, [yD], arc2y, [yF]]).astype(float)

    wire = mlines.Line2D(xs, ys,
                         lw=wire_lw, color=wire_color,
                         solid_capstyle="round",
                         solid_joinstyle="round",
                         zorder=zorder)
    wire.set_clip_path(face_outline)
    ax.add_line(wire)

    # node2 always at the end of the wire
    p2 = (float(xs[-1]), float(ys[-1]))

    node1 = Circle(p1, node_r,
                   facecolor=node_facecolor1,
                   edgecolor=node_edgecolor,
                   linewidth=float(wire_lw) * 0.55,
                   zorder=zorder + 1)
    node1.set_clip_path(face_outline)
    ax.add_patch(node1)

    node2 = Circle(p2, node_r,
                   facecolor=node_facecolor2,
                   edgecolor=node_edgecolor,
                   linewidth=float(wire_lw) * 0.55,
                   zorder=zorder + 1)
    node2.set_clip_path(face_outline)
    ax.add_patch(node2)

    return node1, node2, wire
