from matplotlib.patches import Circle, Arc
import numpy as np


def add_eye(ax, cx, cy,
            open_eye=True,
            r=0.06,
            wink_lw=10,
            outline_color="black",
            iris_color="#49a37d",
            ring_ratio=0.10,
            iris_ratio=0.38,
            pupil_ratio=0.45,
            spec_ratio=0.18,
            spec_offset=(0.25, 0.25),
            iris_offset_y=0.0,
            gaze=(0.0, 0.0),          # NEW: (gx, gy) in [-1, 1] on an imaginary circle
            gaze_strength=0.35,       # NEW: how far pupil/iris can move (fraction of inner radius)
            aa_shrink=0.995,
            zorder=40):

    artists = []
    if open_eye:
        dy = iris_offset_y * r

        sclera = Circle((cx, cy), radius=r, facecolor="white", edgecolor="none", zorder=zorder)
        ax.add_patch(sclera); artists.append(sclera)

        rim = Circle((cx, cy), radius=r, facecolor="none", edgecolor=outline_color,
                     linewidth=wink_lw, zorder=zorder+1)
        ax.add_patch(rim); artists.append(rim)

        inner_r = r * (1.0 - ring_ratio)
        clip_r = inner_r * aa_shrink
        inner_clip = Circle((cx, cy), radius=clip_r, transform=ax.transData)

        # --- NEW: gaze offset in data coords
        gx, gy = gaze
        gx = float(np.clip(gx, -1.0, 1.0))
        gy = float(np.clip(gy, -1.0, 1.0))

        # keep within the unit circle
        norm = (gx*gx + gy*gy) ** 0.5
        if norm > 1.0 and norm > 0:
            gx /= norm
            gy /= norm

        # max offset proportional to inner radius
        max_off = float(gaze_strength) * float(inner_r)
        offx = gx * max_off
        offy = gy * max_off

        ri = inner_r * float(iris_ratio)
        iris = Circle((cx + offx, cy + dy + offy), radius=ri,
                      facecolor=iris_color, edgecolor="none", zorder=zorder+2)
        iris.set_clip_path(inner_clip)
        ax.add_patch(iris); artists.append(iris)

        rp = ri * float(pupil_ratio)
        pupil = Circle((cx + offx, cy + dy + offy), radius=rp,
                       facecolor="black", edgecolor="none", zorder=zorder+3)
        pupil.set_clip_path(inner_clip)
        ax.add_patch(pupil); artists.append(pupil)

        rs = ri * float(spec_ratio)
        ox, oy = spec_offset
        spec = Circle((cx - ox*ri + offx, cy + dy + oy*ri + offy), radius=rs,
                      facecolor="white", edgecolor="none", zorder=zorder+4)
        spec.set_clip_path(inner_clip)
        ax.add_patch(spec); artists.append(spec)

        return artists

    wink = Arc((cx, cy), width=2*r, height=1.15*r,
               angle=0, theta1=200, theta2=340,
               color=outline_color, linewidth=wink_lw,
               capstyle="round", zorder=zorder)
    ax.add_patch(wink); artists.append(wink)
    return artists


def draw_eyes(ax, face_patch,
              cx=0.5, cy=0.5,
              open_left=True, open_right=True,
              eye_dx=0.17,
              eye_dy=-0.05,
              r=0.06,
              wink_lw=10,
              outline_color="black",
              iris_color_left="#49a37d",
              iris_color_right="#49a37d",
              ring_ratio=0.10,
              iris_ratio=0.38,
              pupil_ratio=0.45,
              spec_ratio=0.18,
              spec_offset=(0.30, 0.34),
              iris_offset_y=0.0,
              gaze_left=(0.0, 0.0),       # NEW
              gaze_right=(0.0, 0.0),      # NEW
              gaze_strength=0.35,         # NEW (shared)
              aa_shrink=0.995,
              zorder=40,
              clip=True):

    left_x, left_y = (cx - eye_dx, cy + eye_dy)
    right_x, right_y = (cx + eye_dx, cy + eye_dy)

    left_artists = add_eye(
        ax, left_x, left_y, open_eye=open_left,
        r=r, wink_lw=wink_lw,
        outline_color=outline_color,
        iris_color=iris_color_left,
        ring_ratio=ring_ratio,
        iris_ratio=iris_ratio,
        pupil_ratio=pupil_ratio,
        spec_ratio=spec_ratio,
        spec_offset=spec_offset,
        iris_offset_y=iris_offset_y,
        gaze=gaze_left,
        gaze_strength=gaze_strength,
        aa_shrink=aa_shrink,
        zorder=zorder
    )

    right_artists = add_eye(
        ax, right_x, right_y, open_eye=open_right,
        r=r, wink_lw=wink_lw,
        outline_color=outline_color,
        iris_color=iris_color_right,
        ring_ratio=ring_ratio,
        iris_ratio=iris_ratio,
        pupil_ratio=pupil_ratio,
        spec_ratio=spec_ratio,
        spec_offset=spec_offset,
        iris_offset_y=iris_offset_y,
        gaze=gaze_right,
        gaze_strength=gaze_strength,
        aa_shrink=aa_shrink,
        zorder=zorder
    )

    if clip:
        for a in left_artists + right_artists:
            a.set_clip_path(face_patch)

    return left_artists, right_artists
