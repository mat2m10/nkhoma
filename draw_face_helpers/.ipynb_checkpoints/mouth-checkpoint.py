from matplotlib.patches import Arc

# ----------------------------
# Smile
# ----------------------------
def add_smile(ax, face_patch,
              cx=0.5, cy=0.5,
              y=-0.18,
              w=0.42,
              h=0.26,
              theta1=200, theta2=340,
              color="black",
              lw=10,
              zorder=45,
              clip=True):
    smile = Arc((cx, cy + y), width=w, height=h,
                angle=0, theta1=theta1, theta2=theta2,
                color=color, linewidth=lw, capstyle="round",
                zorder=zorder)
    if clip:
        smile.set_clip_path(face_patch)
    ax.add_patch(smile)
    return smile
