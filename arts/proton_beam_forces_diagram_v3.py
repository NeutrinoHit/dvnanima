#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib import patheffects as pe

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"

p1 = np.array([-1.45, 0.0])
p2 = np.array([ 1.45, 0.0])
R = np.linalg.norm(p2 - p1)

red = "#d21f1f"
red_dark = "#7c1010"
green = "#16833a"
blue1 = "#0d4da1"
blue2 = "#1a67c7"
gray = "#6e6e6e"
black = "#111111"

fig, ax = plt.subplots(figsize=(10, 10), dpi=220)
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-5.4, 5.4)
ax.set_ylim(-5.25, 5.25)

def stroke_text(txt, lw=3.5):
    txt.set_path_effects([pe.withStroke(linewidth=lw, foreground="white")])
    return txt

ax.add_patch(Circle((0, -0.05), 4.55, color="#f7f7fb", zorder=-10))
ax.add_patch(Circle((0, -0.05), 4.15, fill=False, lw=0.8, color="#e1e4ef", zorder=-9))

def circle_with_tangent_arrows(center, radius, color, label, label_xy, thetas, lw=3.0):
    th = np.linspace(0, 2*np.pi, 900)
    ax.plot(center[0] + radius*np.cos(th), center[1] + radius*np.sin(th),
            color=color, lw=lw, zorder=1)

    for theta in thetas:
        dt = 0.13
        t1, t2 = theta + dt, theta - dt  # clockwise
        a = center + radius*np.array([np.cos(t1), np.sin(t1)])
        b = center + radius*np.array([np.cos(t2), np.sin(t2)])
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18,
                                     lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=3))
    stroke_text(ax.text(*label_xy, label, fontsize=19, fontweight="bold",
                        color=color, ha="center", va="center", zorder=9))

# B1 centered on proton 1; B2 centered on proton 2; both clockwise for v into screen
circle_with_tangent_arrows(p1, R, blue1, r"$\mathbf{B}_1$", (-3.1, 2.55),
                           thetas=[0.03, 2.35, 5.45])
circle_with_tangent_arrows(p2, R, blue2, r"$\mathbf{B}_2$", (3.1, 2.55),
                           thetas=[np.pi-0.03, 0.78, 3.95])

def velocity_line(p):
    start = p + np.array([0.0, 0.48])
    end = p + np.array([0.72, 3.15])
    ax.plot([start[0], end[0]], [start[1], end[1]],
            linestyle=(0, (6, 6)), lw=2.0, color=gray, zorder=0)
    ax.add_patch(FancyArrowPatch(start + 0.62*(end-start), end,
                                 arrowstyle="-|>", mutation_scale=16,
                                 lw=2.0, linestyle=(0, (6, 6)),
                                 color=gray, zorder=4))
    stroke_text(ax.text(end[0]+0.18, end[1]+0.15, r"$\mathbf{v}$",
                        fontsize=20, color=black, fontweight="bold",
                        ha="center", va="center", zorder=10))

velocity_line(p1)
velocity_line(p2)

ax.plot([p1[0] + 0.55, p2[0] - 0.55], [0, 0],
        color="#999999", lw=1.5, linestyle=(0, (4, 5)), zorder=0)
stroke_text(ax.text(0, -0.24, r"$r$", fontsize=18,
                    ha="center", va="top", color=black, zorder=8))

def force_arrow(start, end, color, label, label_xy, fontsize=20):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=21,
                                 lw=4.6, color=color, shrinkA=0, shrinkB=0, zorder=8))
    stroke_text(ax.text(*label_xy, label, fontsize=fontsize, fontweight="bold",
                        color=color, ha="center", va="center", zorder=11))

# All forces lie on the horizontal diameter
force_arrow(p1 + np.array([-0.55, 0.0]), p1 + np.array([-1.42, 0.0]),
            red, r"$\mathbf{F}_e$", p1 + np.array([-1.05, 0.42]))
force_arrow(p2 + np.array([0.55, 0.0]), p2 + np.array([1.42, 0.0]),
            red, r"$\mathbf{F}_e$", p2 + np.array([1.05, 0.42]))
force_arrow(p1 + np.array([0.55, 0.0]), p1 + np.array([1.26, 0.0]),
            green, r"$\mathbf{F}_m$", p1 + np.array([1.02, -0.45]))
force_arrow(p2 + np.array([-0.55, 0.0]), p2 + np.array([-1.26, 0.0]),
            green, r"$\mathbf{F}_m$", p2 + np.array([-1.02, -0.45]))

def draw_proton(center, label):
    ax.add_patch(Circle(center + np.array([0.10, -0.10]), 0.45, color="black", alpha=0.16, zorder=5))
    for i, rr in enumerate(np.linspace(0.42, 0.02, 26)):
        t = i / 25
        col = ((1-t)*0.72 + t*1.00, (1-t)*0.05 + t*0.55, (1-t)*0.05 + t*0.55)
        ax.add_patch(Circle(center + np.array([-0.09*t, 0.11*t]), rr, color=col, ec=None, zorder=6))
    ax.add_patch(Circle(center, 0.42, fill=False, lw=2.0, color=red_dark, zorder=9))
    plus = ax.text(center[0], center[1]+0.01, "+", fontsize=35, fontweight="bold",
                   color="white", ha="center", va="center", zorder=12)
    plus.set_path_effects([pe.withStroke(linewidth=2.0, foreground=red_dark)])
    stroke_text(ax.text(center[0], center[1]-0.68, label, fontsize=13,
                        color=black, ha="center", va="center", zorder=12), lw=2.7)

draw_proton(p1, "протон 1")
draw_proton(p2, "протон 2")

stroke_text(ax.text(0, 4.75, "Силы между двумя протонами в пучке",
                    fontsize=22, fontweight="bold", ha="center", color=black, zorder=20))

# Formula block below magnetic-field circles. Circles bottom at y=-R=-2.9.
box_y = -3.62
box = FancyBboxPatch((-2.85, box_y - 0.38), 5.70, 0.90,
                     boxstyle="round,pad=0.12,rounding_size=0.12",
                     fc="white", ec="#d8d8e5", lw=1.2, alpha=0.94, zorder=14)
ax.add_patch(box)
ax.text(0, box_y + 0.05,
        r"$|\mathbf{F}_{\rm res}|\simeq F_e(1-\beta^2)=F_e/\gamma^2$",
        fontsize=20, ha="center", va="center", color=black, zorder=20)

ax.text(-4.65, -4.55, r"$\mathbf{F}_e$ — электрическое отталкивание",
        fontsize=13.5, color=red, ha="left", zorder=20)
ax.text(-4.65, -4.90, r"$\mathbf{F}_m$ — магнитное притяжение",
        fontsize=13.5, color=green, ha="left", zorder=20)
ax.text(0.90, -4.55, r"$\mathbf{B}_1,\mathbf{B}_2$ — магнитные поля протонов",
        fontsize=13.5, color=blue1, ha="left", zorder=20)
ax.text(0.90, -4.90, r"$\mathbf{v}$ — скорость протонов",
        fontsize=13.5, color=black, ha="left", zorder=20)

# Save without tight cropping; this avoids local renderer/font-dependent clipping.
plt.savefig("proton_beam_forces_diagram_v3.png", dpi=220, facecolor="white")
plt.savefig("proton_beam_forces_diagram_v3.svg", facecolor="white")
plt.close(fig)
