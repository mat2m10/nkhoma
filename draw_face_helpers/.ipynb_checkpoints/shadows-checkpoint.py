import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# ----------------------------
# Face-follow crescent shading using x_center +/- n gating
# ----------------------------
def add_face_follow_crescent(ax, face_patch,
                             side="right",
                             color="#5b2cff",
                             alpha=0.22,
                             width_frac=0.10,
                             margin_frac=0.06,
                             smooth_window=11,
                             zorder=28):
    path = face_patch.get_path().transformed(face_patch.get_patch_transform())
    dense = path.interpolated(300).to_polygons(closed_only=True)
    if not dense:
        return None
    poly = max(dense, key=len)

    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    W = float(xmax - xmin)

    x_center = 0.5 * (xmin + xmax)
    n = float(margin_frac) * W

    side = side.lower().strip()
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    if side == "right":
        mask = poly[:, 0] > (x_center + n)
        inward_sign = -1.0







def add_circle_lens_shadow(ax,
                           face_outline,
                           circle_center=(0.5, 0.5),
                           radius=1.25,
                           brain_shadow_color="#5b2cff",
                           face_shadow_color="#5b2cff",
                           alpha=0.18,
                           poly_interp=600,
                           circle_n=1200,
                           zorder=28,
                           region="outside",   # "inside" or "outside"
                           prefer="right"):    # "right" or "left"
    """
    region="inside"  -> shades face ∩ circle  (your previous behavior)
    region="outside" -> shades face \ circle  (the one you want, to the right in your example)
    prefer selects which side if there are multiple candidate regions.
    """

    # ---- dense face polygon (data coords)
    path_data = face_outline.get_path().transformed(face_outline.get_patch_transform())
    poly_list = path_data.interpolated(int(poly_interp)).to_polygons(closed_only=True)
    if not poly_list:
        return None, None
    face_poly = max(poly_list, key=len)
    face_path = Path(face_poly)

    cx, cy = map(float, circle_center)
    r = float(radius)

    # ---- sample points on circle
    ang = np.linspace(0.0, 2.0 * np.pi, int(circle_n), endpoint=False)
    circle_pts = np.c_[cx + r * np.cos(ang), cy + r * np.sin(ang)]

    # points on circle that lie inside the face (this is the circle arc used as boundary)
    inside_face = face_path.contains_points(circle_pts)

    # ---- helper: all contiguous True runs on a circular boolean array
    def all_true_runs(mask):
        mask = np.asarray(mask, dtype=bool)
        n = len(mask)
        if n == 0 or not mask.any():
            return []

        # find transitions
        m = mask.astype(int)
        # indices where value changes
        changes = np.where(np.diff(np.r_[m, m[0]]) != 0)[0] + 1

        if len(changes) == 0:
            return [np.arange(n)]

        # build segments between changes
        segs = []
        start = changes[0]
        for end in changes[1:]:
            seg = np.arange(start, end) % n
            if mask[seg[0]]:
                segs.append(seg)
            start = end
        seg = np.arange(start, changes[0] + n) % n
        if mask[seg[0]]:
            segs.append(seg)

        # keep only “real” segments
        return [s for s in segs if len(s) >= 3]

    # circle arc candidates (usually 1, but keep it general)
    arc_runs = all_true_runs(inside_face)
    if not arc_runs:
        return None, None

    # ---- face boundary classification: inside vs outside circle
    d2 = (face_poly[:, 0] - cx) ** 2 + (face_poly[:, 1] - cy) ** 2
    inside_circle = d2 <= (r * r)

    if region.lower().strip() == "inside":
        face_mask = inside_circle
    elif region.lower().strip() == "outside":
        face_mask = ~inside_circle
    else:
        raise ValueError("region must be 'inside' or 'outside'")

    face_runs = all_true_runs(face_mask)
    if not face_runs:
        return None, None

    # ---- choose the best (face_run, arc_run) pair by side preference
    face_center_x = float(0.5 * (face_poly[:, 0].min() + face_poly[:, 0].max()))
    want_right = (prefer.lower().strip() == "right")

    best = None
    best_score = None

    for fr in face_runs:
        face_edge_pts = face_poly[fr]
        A = face_edge_pts[0]
        B = face_edge_pts[-1]

        for ar in arc_runs:
            arc_pts = circle_pts[ar]

            # pick arc direction that matches endpoints better
            s1 = np.linalg.norm(arc_pts[0] - A) + np.linalg.norm(arc_pts[-1] - B)
            s2 = np.linalg.norm(arc_pts[-1] - A) + np.linalg.norm(arc_pts[0] - B)
            arc_use = arc_pts if (s1 <= s2) else arc_pts[::-1]

            # stitch polygon: face segment + reversed arc (to close)
            pts = np.vstack([face_edge_pts, arc_use[::-1]])

            # score by centroid-x (prefer rightmost or leftmost)
            cx_poly = float(np.mean(pts[:, 0]))
            score = (cx_poly - face_center_x)
            score = score if want_right else (-score)

            if (best is None) or (score > best_score):
                best = pts
                best_score = score

    if best is None or len(best) < 10:
        return None, None

    lens_pts = best

    # ---- build patch from stitched polygon
    verts = [tuple(lens_pts[0])]
    codes = [Path.MOVETO]
    for p in lens_pts[1:]:
        verts.append(tuple(p))
        codes.append(Path.LINETO)
    verts.append((0.0, 0.0))
    codes.append(Path.CLOSEPOLY)
    lens_path = Path(verts, codes)

    # draw twice, clip separately so you can color brain vs face differently
    brain_shadow = PathPatch(lens_path, facecolor=brain_shadow_color, edgecolor="none",
                             alpha=alpha, zorder=zorder)
    brain_shadow.set_clip_path(face_outline)
    ax.add_patch(brain_shadow)

    face_shadow = PathPatch(lens_path, facecolor=face_shadow_color, edgecolor="none",
                            alpha=alpha, zorder=zorder)
    face_shadow.set_clip_path(face_outline)
    ax.add_patch(face_shadow)

    return brain_shadow, face_shadow
