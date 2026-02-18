# ----------------------------
# px->data conversions
# ----------------------------

import numpy as np
from matplotlib.patches import Circle



def background_circle(ax,
                      center=(0.5, 0.5),
                      radius=0.75,
                      color="#EEF2FF",
                      alpha=1.0,
                      edgecolor="none",
                      lw=0.0,
                      zorder=0):
    """
    Draw a simple background circle behind everything.
    You fully control the radius.

    Returns the Circle patch.
    """
    cx, cy = map(float, center)
    r = float(radius)

    bg = Circle(
        (cx, cy),
        radius=r,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=float(lw),
        alpha=float(alpha),
        zorder=int(zorder),
    )
    ax.add_patch(bg)
    return bg





def _px_to_data_dy(ax, px):
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0.0, 0.0))
    x1, y1 = inv.transform((0.0, float(px)))
    return (y1 - y0)

def _px_to_data_dx(ax, px):
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0.0, 0.0))
    x1, y1 = inv.transform((float(px), 0.0))
    return (x1 - x0)