#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib import patheffects as pe

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"

# -----------------------------
# Geometry
# -----------------------------
p1 = np.array([-1.55, 1.28])
p2 = np.array([ 1.55, 1.28])
R = np.linalg.norm(p2 - p1)

# -----------------------------
# Colors
# -----------------------------
red = "#d21f1f"
red_dark = "#7c1010"
green = "#16833a"
blue1 = "#0d4da1"
blue2 = "#1a67c7"
gray = "#6e6e6e"
black = "#111111"

# -----------------------------
# Figure
# -----------------------------
fig, ax = plt.subplots(figsize=(8.2, 8.2), dpi=220)
ax.set_aspect("equal")
ax.axis("off")

# Чуть расширяем вниз, чтобы спокойно опустить блок с формулами
ax.set_xlim(-5.35, 5.35)
ax.set_ylim(-5.65, 5.35)

def stroke_text(txt, lw=2.6):
    txt.set_path_effects([pe.withStroke(linewidth=lw, foreground="white")])
    return txt

# -----------------------------
# Background
# -----------------------------
ax.add_patch(Circle((0, 0.70), 4.95, color="#f7f7fb", zorder=-10))
ax.add_patch(Circle((0, 0.70), 4.65, fill=False, lw=0.8, color="#e1e4ef", zorder=-9))

# -----------------------------
# Magnetic field circles
# -----------------------------
def circle_with_tangent_arrows(center, radius, color, label, label_xy, thetas, lw=2.9):
    th = np.linspace(0, 2*np.pi, 900)
    ax.plot(center[0] + radius*np.cos(th),
            center[1] + radius*np.sin(th),
            color=color, lw=lw, zorder=1)

    # v направлена вглубь экрана; B-окружности по часовой стрелке в плоскости рисунка
    for theta in thetas:
        dt = 0.13
        t1, t2 = theta + dt, theta - dt
        a = center + radius*np.array([np.cos(t1), np.sin(t1)])
        b = center + radius*np.array([np.cos(t2), np.sin(t2)])
        ax.add_patch(FancyArrowPatch(
            a, b,
            arrowstyle="-|>",
            mutation_scale=16.5,
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=3
        ))

    stroke_text(ax.text(*label_xy, label,
                        fontsize=17.5,
                        fontweight="bold",
                        color=color,
                        ha="center",
                        va="center",
                        zorder=9))

# B1 centered on proton 1 and passing through proton 2
circle_with_tangent_arrows(
    p1, R, blue1, r"$\mathbf{B}_1$", (-3.25, 3.45),
    thetas=[0.03, 2.28, 5.40]
)

# B2 centered on proton 2 and passing through proton 1
circle_with_tangent_arrows(
    p2, R, blue2, r"$\mathbf{B}_2$", (3.25, 3.45),
    thetas=[np.pi - 0.03, 0.90, 3.98]
)

# -----------------------------
# Velocity lines: perspective
# -----------------------------
vp = np.array([0.0, 4.0])  # общая точка схода

def velocity_line(p, frac=0.55):
    start = p + np.array([0.0, 0.48])
    end = start + frac * (vp - start)

    ax.plot([start[0], end[0]], [start[1], end[1]],
            linestyle=(0, (6, 6)),
            lw=1.9,
            color=gray,
            zorder=0)

    ax.add_patch(FancyArrowPatch(
        start + 0.62 * (end - start),
        end,
        arrowstyle="-|>",
        mutation_scale=14.5,
        lw=1.9,
        linestyle=(0, (6, 6)),
        color=gray,
        zorder=4
    ))

    dx = 0.10 if end[0] >= 0 else -0.10
    stroke_text(ax.text(end[0] + dx, end[1] + 0.10,
                        r"$\mathbf{v}$",
                        fontsize=17.5,
                        color=black,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        zorder=10))

velocity_line(p1)
velocity_line(p2)

# -----------------------------
# Distance line
# -----------------------------
ax.plot([p1[0] + 0.52, p2[0] - 0.52],
        [p1[1], p1[1]],
        color="#999999",
        lw=1.4,
        linestyle=(0, (4, 5)),
        zorder=0)

stroke_text(ax.text(0, p1[1] - 0.22,
                    r"$r$",
                    fontsize=16,
                    ha="center",
                    va="top",
                    color=black,
                    zorder=8))

# -----------------------------
# Forces
# -----------------------------
def force_arrow(start, end, color, label, label_xy, fontsize=18.5):
    ax.add_patch(FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=19,
        lw=4.1,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=8
    ))

    stroke_text(ax.text(*label_xy,
                        label,
                        fontsize=fontsize,
                        fontweight="bold",
                        color=color,
                        ha="center",
                        va="center",
                        zorder=11))

# Electric repulsion: outward
force_arrow(p1 + np.array([-0.52, 0.0]),
            p1 + np.array([-1.34, 0.0]),
            red,
            r"$\mathbf{F}_e$",
            p1 + np.array([-0.98, 0.36]))

force_arrow(p2 + np.array([0.52, 0.0]),
            p2 + np.array([1.34, 0.0]),
            red,
            r"$\mathbf{F}_e$",
            p2 + np.array([0.98, 0.36]))

# Magnetic attraction: inward
force_arrow(p1 + np.array([0.52, 0.0]),
            p1 + np.array([1.18, 0.0]),
            green,
            r"$\mathbf{F}_m$",
            p1 + np.array([0.94, -0.40]))

force_arrow(p2 + np.array([-0.52, 0.0]),
            p2 + np.array([-1.18, 0.0]),
            green,
            r"$\mathbf{F}_m$",
            p2 + np.array([-0.94, -0.40]))

# -----------------------------
# Protons
# -----------------------------
def draw_proton(center, label):
    ax.add_patch(Circle(center + np.array([0.09, -0.09]),
                        0.41,
                        color="black",
                        alpha=0.15,
                        zorder=5))

    for i, rr in enumerate(np.linspace(0.38, 0.02, 24)):
        t = i / 23
        col = ((1 - t) * 0.72 + t * 1.00,
               (1 - t) * 0.05 + t * 0.55,
               (1 - t) * 0.05 + t * 0.55)

        ax.add_patch(Circle(center + np.array([-0.08 * t, 0.10 * t]),
                            rr,
                            color=col,
                            ec=None,
                            zorder=6))

    ax.add_patch(Circle(center,
                        0.38,
                        fill=False,
                        lw=1.9,
                        color=red_dark,
                        zorder=9))

    plus = ax.text(center[0],
                   center[1] + 0.01,
                   "+",
                   fontsize=31,
                   fontweight="bold",
                   color="white",
                   ha="center",
                   va="center",
                   zorder=12)

    plus.set_path_effects([pe.withStroke(linewidth=1.8, foreground=red_dark)])

    stroke_text(ax.text(center[0],
                        center[1] - 0.60,
                        label,
                        fontsize=12.0,
                        color=black,
                        ha="center",
                        va="center",
                        zorder=12),
                lw=2.4)

draw_proton(p1, "протон 1")
draw_proton(p2, "протон 2")

# -----------------------------
# Title
# -----------------------------
stroke_text(ax.text(0, 5.65,
                    "Силы между двумя протонами в пучке",
                    fontsize=20.5,
                    fontweight="bold",
                    ha="center",
                    color=black,
                    zorder=20))

# -----------------------------
# Compact derivation panel
# -----------------------------
panel = FancyBboxPatch(
    (-4.25, -5.28),
    8.50,
    2.05,
    boxstyle="round,pad=0.16,rounding_size=0.12",
    fc="white",
    ec="#d8d8e5",
    lw=1.2,
    alpha=0.97,
    zorder=13
)
ax.add_patch(panel)

stroke_text(ax.text(-3.95, -3.24,
                    "Быстрый вывод",
                    fontsize=14.5,
                    fontweight="bold",
                    color=black,
                    ha="left",
                    va="center",
                    zorder=20),
            lw=2.2)

lines = [
    r"$\mathbf{F}=q(\mathbf{E}+\mathbf{v}\times\mathbf{B}),\qquad \mathbf{B}=\mathbf{v}\times\mathbf{E}$",
    r"$F_m=qvB=qE\,v^2=v^2F_e$",
    r"$F_{\rm res}=F_e-F_m=F_e(1-v^2)=\frac{F_e}{\gamma^2}, \qquad \gamma=\frac{1}{\sqrt{1-v^2}}$",
]

ys = [-3.82, -4.43, -5.03]
sizes = [14.1, 14.1, 14.1]

for y, s, line in zip(ys, sizes, lines):
    ax.text(-3.95,
            y,
            line,
            fontsize=s,
            color=black,
            ha="left",
            va="center",
            zorder=20)

# -----------------------------
# Save
# -----------------------------
plt.savefig("proton_beam_forces_diagram.png", dpi=220, facecolor="white")
plt.savefig("proton_beam_forces_diagram.svg", facecolor="white")
plt.close(fig)