
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib import patheffects as pe

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"

fig, ax = plt.subplots(figsize=(10, 10), dpi=220)
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-4.9, 4.9)
ax.set_ylim(-4.6, 5.3)

p1 = np.array([-1.45, 0.0])
p2 = np.array([ 1.45, 0.0])
R = np.linalg.norm(p2 - p1)

red = "#d21f1f"
red_dark = "#7c1010"
green = "#16833a"
blue1 = "#0d4da1"
blue2 = "#1a67c7"
gray = "#707070"
black = "#111111"

ax.add_patch(Circle((0, 0.2), 4.7, color="#f7f7fb", zorder=-10))
ax.add_patch(Circle((0, 0.2), 4.3, fill=False, lw=0.8, color="#e2e5ef", zorder=-9))

def stroke_text(txt, lw=4):
    txt.set_path_effects([pe.withStroke(linewidth=lw, foreground="white")])
    return txt

def circle_with_arrow(center, radius, color, label, label_xy, arrow_thetas):
    th = np.linspace(0, 2*np.pi, 800)
    x = center[0] + radius*np.cos(th)
    y = center[1] + radius*np.sin(th)
    ax.plot(x, y, color=color, lw=3.0, zorder=1)

    for theta in arrow_thetas:
        dt = 0.14
        t1, t2 = theta + dt, theta - dt  # clockwise
        a = center + radius*np.array([np.cos(t1), np.sin(t1)])
        b = center + radius*np.array([np.cos(t2), np.sin(t2)])
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=20,
            lw=3.0, color=color, shrinkA=0, shrinkB=0, zorder=3
        ))

    stroke_text(ax.text(*label_xy, label, fontsize=23, fontweight="bold",
                        color=color, ha="center", va="center", zorder=8))

circle_with_arrow(p1, R, blue1, r"$\mathbf{B}_1$", (-3.0, 2.60), arrow_thetas=[2.25, 5.55])
circle_with_arrow(p2, R, blue2, r"$\mathbf{B}_2$", ( 3.0, 2.60), arrow_thetas=[0.90, 3.95])

def velocity_line(p):
    start = p + np.array([0.0, 0.38])
    end = p + np.array([0.78, 3.35])
    ax.plot([start[0], end[0]], [start[1], end[1]],
            linestyle=(0, (6, 6)), lw=2.0, color=gray, zorder=0)
    ax.add_patch(FancyArrowPatch(
        start + 0.60*(end-start), end,
        arrowstyle="-|>", mutation_scale=18,
        lw=2.0, linestyle=(0, (6, 6)),
        color=gray, zorder=4
    ))
    stroke_text(ax.text(end[0] + 0.14, end[1] + 0.10, r"$\mathbf{v}$",
                        fontsize=23, color=black, ha="center", va="center", zorder=8))

velocity_line(p1)
velocity_line(p2)

ax.plot([p1[0], p2[0]], [0, 0], color="#9a9a9a", lw=1.6,
        linestyle=(0, (4, 5)), zorder=0)
stroke_text(ax.text(0, -0.26, r"$r$", fontsize=20, ha="center", va="top", color=black, zorder=8))

def force_arrow(start, end, color, label, label_xy):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=24,
        lw=5.0, color=color, zorder=7
    ))
    stroke_text(ax.text(*label_xy, label, fontsize=23, fontweight="bold",
                        color=color, ha="center", va="center", zorder=9))

y_red = 0.00
y_green = 0.00

force_arrow(np.array([p1[0] - 0.25, y_red]), np.array([p1[0] - 1.25, y_red]), red,
            r"$\mathbf{F}_e$", np.array([p1[0] - 0.90, y_red + 0.38]))
force_arrow(np.array([p2[0] + 0.25, y_red]), np.array([p2[0] + 1.25, y_red]), red,
            r"$\mathbf{F}_e$", np.array([p2[0] + 0.90, y_red + 0.38]))

force_arrow(np.array([p1[0] + 0.25, y_green]), np.array([p1[0] + 1.08, y_green]), green,
            r"$\mathbf{F}_m$", np.array([p1[0] + 0.86, y_green - 0.42]))
force_arrow(np.array([p2[0] - 0.25, y_green]), np.array([p2[0] - 1.08, y_green]), green,
            r"$\mathbf{F}_m$", np.array([p2[0] - 0.86, y_green - 0.42]))

def proton(center, label):
    ax.add_patch(Circle(center + np.array([0.09, -0.10]), 0.47, color="black", alpha=0.15, zorder=5))
    for i, rr in enumerate(np.linspace(0.45, 0.03, 28)):
        t = i / 27
        color = ((1-t)*0.72 + t*1.00, (1-t)*0.05 + t*0.55, (1-t)*0.05 + t*0.55)
        ax.add_patch(Circle(center + np.array([-0.10*t, 0.12*t]), rr, color=color, ec=None, zorder=6))
    ax.add_patch(Circle(center, 0.45, fill=False, lw=2.0, color=red_dark, zorder=8))
    plus = ax.text(center[0], center[1], "+", fontsize=38, fontweight="bold",
                   color="white", ha="center", va="center", zorder=10)
    plus.set_path_effects([pe.withStroke(linewidth=2.2, foreground=red_dark)])
    stroke_text(ax.text(center[0], center[1] - 0.72, label, fontsize=16,
                        color=black, ha="center", va="center", zorder=10), lw=3)

proton(p1, "протон 1")
proton(p2, "протон 2")

stroke_text(ax.text(0, 4.78, "Силы между двумя протонами в пучке",
                    fontsize=25, fontweight="bold",
                    ha="center", color=black, zorder=20))

box_y = -3.15
box = FancyBboxPatch(
    (-2.6, box_y - 0.35), 5.2, 0.95,
    boxstyle="round,pad=0.12,rounding_size=0.12",
    fc="white", ec="#d9d9e7", lw=1.2, alpha=0.95, zorder=11
)
ax.add_patch(box)
ax.text(0, box_y + 0.08,
        r"$|\mathbf{F}_{\rm res}|\simeq F_e(1-\beta^2)=F_e/\gamma^2$",
        fontsize=23, ha="center", va="center", color=black, zorder=20)

ax.text(-4.25, -4.05, r"$\mathbf{F}_e$ — электрическое отталкивание",
        fontsize=15, color=red, ha="left")
ax.text(-4.25, -4.38, r"$\mathbf{F}_m$ — магнитное притяжение",
        fontsize=15, color=green, ha="left")
ax.text(1.0, -4.05, r"$\mathbf{B}_1,\mathbf{B}_2$ — магнитные поля протонов",
        fontsize=15, color=blue1, ha="left")
ax.text(1.0, -4.38, r"$\mathbf{v}$ — скорость протонов",
        fontsize=15, color=black, ha="left")

plt.savefig("proton_beam_forces_diagram_v2.png", bbox_inches="tight", pad_inches=0.12, dpi=220)
plt.savefig("proton_beam_forces_diagram_v2.svg", bbox_inches="tight", pad_inches=0.12)
plt.show()
