r"""Circular polarization: the rotation of E, the helicity of the photon and how linear and elliptical light are built.

A plane wave goes along +z.  The two circular polarizations of the book (the naming of particle physics: the handedness
is the sign of the helicity)

    eps_R = (1,  i, 0)/sqrt2   ->  E_R(t, z) = (cos(wt - kz),  sin(wt - kz), 0),   photon spin projection S_z = +hbar,
    eps_L = (1, -i, 0)/sqrt2   ->  E_L(t, z) = (cos(wt - kz), -sin(wt - kz), 0),   photon spin projection S_z = -hbar,

so at a fixed z the vector E_R rotates clockwise for an observer at the source looking along the wave (this defines
"right" in the book) and counterclockwise for an observer to whom the wave comes (the receiver).  This is positive
helicity: the rotation of E is a right-handed screw about k.  Optics names the same wave left-circular, because it
counts from the receiver's view.

A snapshot of the same wave in space (fixed t) is a *left-handed* helix, the tip of E_R at z has the azimuth
wt - kz which decreases with z.  The film shows both the snapshot and the rotation in time.

Any polarization is a combination eps = a_R eps_R + a_L eps_L.  With |a_R| != |a_L| and a relative phase delta it is
an ellipse with the semi-axes |a_R| + |a_L| and ||a_R| - |a_L||, tilted by -delta/2; |a_R| = |a_L| is a linear
polarization along the direction -delta/2.

Film time (s), 52 s: 0-15 positive helicity (E_R) in 3D with the two views; 15-28 negative helicity (E_L);
28-47 E_R + E_L: linear, tilted, elliptical, circular; 47-52 photon spin.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python circular_polarization.py --lang en            # film -> media/circular_polarization_en.mp4
    python circular_polarization.py --lang ru
    python circular_polarization.py --lang en --snapshot 10
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config, load_texts  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)

TOTAL = CFG.timeline.total
OMEGA = 2 * math.pi / CFG.wave.period          # film angular frequency of the rotation
KWAVE = 2 * math.pi / CFG.wave.wavelength      # wave number of the 3D snapshot
L_Z = CFG.wave.length                          # length of the 3D wave


# ------------------------------------------------------------------------ physics

def field_circular(t: float, z: np.ndarray | float, helicity: int) -> np.ndarray:
    """E(t, z) of a circular wave, helicity +1 -> eps_R (1, i, 0), -1 -> eps_L (1, -i, 0); shape (..., 3)."""
    ph = OMEGA * t - KWAVE * np.asarray(z, float)
    return np.stack([np.cos(ph), helicity * np.sin(ph), np.zeros_like(ph)], axis=-1)


def field_from_amplitudes(t: float, a_l: float, a_r: float, delta: float) -> np.ndarray:
    """E(t) at z = 0 of a_L E_L + a_R E_R with the relative phase delta (a_R carries e^{i delta} in the complex notation)."""
    x = a_l * math.cos(OMEGA * t) + a_r * math.cos(OMEGA * t + delta)
    y = a_l * math.sin(OMEGA * t) - a_r * math.sin(OMEGA * t + delta)
    return np.array([x, y, 0.0])


def ellipse_axes(a_l: float, a_r: float) -> tuple[float, float]:
    """Semi-axes (major, minor) of the polarization ellipse."""
    return a_l + a_r, abs(a_l - a_r)


def ellipse_tilt(delta: float) -> float:
    """Direction of the major axis (radians, modulo pi)."""
    return -0.5 * delta


def angular_velocity_sign(helicity: int, view: str) -> int:
    """+1 counterclockwise, -1 clockwise for the view 'receiver' (wave comes to the observer) or 'source'."""
    s = helicity                       # E_L (helicity +1) is counterclockwise at the receiver
    return s if view == "receiver" else -s


def poynting_z(helicity: int) -> float:
    return 1.0


def spin_projection(helicity: int) -> int:
    return helicity


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


# amplitude schedule of the third part -------------------------------------------------

def mix_parameters(t: float) -> tuple[float, float, float]:
    """(a_L, a_R, delta) in the part 28-47 s: linear -> tilted -> elliptical -> circular."""
    M_ = CFG.mix
    s = 1 / math.sqrt(2)
    if t < M_.start:
        return s, s, 0.0
    # linear along x, then the axis turns (delta), then a_R shrinks to 0 (ellipse -> circle)
    delta = M_.delta_max * smooth(t, *M_.delta_ramp)
    ar = s * (1.0 - smooth(t, *M_.shrink_ramp))
    return s, ar, delta


TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def num(x: float, fmt: str, lang: str) -> str:
    """A number for use inside $...$: decimal comma in Russian."""
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, CA, S3, PN, MX, RD, PT, LY, ST, TLN = (CFG.video, CFG.camera, CFG.scene3d, CFG.panels, CFG.mixing, CFG.readouts, CFG.panel_text, CFG.layout,
                                              CFG.style, CFG.timeline)
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = ST.background
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)
    TXT = tuple(ST.text)
    DIM = tuple(ST.dim)
    C_POS = tuple(ST.positive)
    C_NEG = tuple(ST.negative)
    C_SUM = tuple(ST.sum)
    EDGE = tuple(ST.panel_edge)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    d3 = fig.add_axes(S3.axes, facecolor="none")
    v1 = fig.add_axes(PN.view1_axes, facecolor="none")
    v2 = fig.add_axes(PN.view2_axes, facecolor="none")
    big = fig.add_axes(PN.big_axes, facecolor="none")
    SA = PN.spin_axes
    spin_axes = [fig.add_axes([SA["x0"] + SA["dx"] * i, SA["y"], SA["w"], SA["h"]], facecolor="none") for i in range(2)]
    k = total / TOTAL
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    phi_c, th_c = math.radians(CA.phi_deg), math.radians(CA.theta_deg)

    def proj(x, y, z):
        """Orthographic view of a right-handed frame: x up, z to the right and a little up, y towards the viewer."""
        x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
        u = z * math.cos(phi_c) - y * math.sin(phi_c)
        v = x * math.cos(th_c) + (z * math.sin(phi_c) + y * math.cos(phi_c)) * math.sin(th_c)
        return u, v

    def style_small(ax):
        ax.set_facecolor("none")
        ax.set_xlim(-PN.lim, PN.lim)
        ax.set_ylim(-PN.lim, PN.lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    def view_panel(ax, helicity, view, tcur, color, title):
        ax.clear()
        style_small(ax)
        a = np.linspace(0, 2 * math.pi, PN.circle_points)
        ax.plot(np.cos(a), np.sin(a), color=(*EDGE, PN.circle_alpha), lw=PN.circle_width * sc)
        ax.plot([-PN.cross_half, PN.cross_half], [0, 0], color=(1, 1, 1, PN.cross_alpha), lw=PN.cross_width * sc)
        ax.plot([0, 0], [-PN.cross_half, PN.cross_half], color=(1, 1, 1, PN.cross_alpha), lw=PN.cross_width * sc)
        e = field_circular(tcur, 0.0, helicity)
        x, y = float(e[0]), float(e[1])
        if view == "source":
            x = -x                              # looking along +z: the x axis points to the left
        # the trail of the tip
        tr = np.linspace(tcur - PN.trail_seconds, tcur, PN.trail_points)
        pts = np.array([field_circular(tt_, 0.0, helicity)[:2] for tt_ in tr])
        if view == "source":
            pts[:, 0] = -pts[:, 0]
        for i in range(len(pts) - 1):
            ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1], color=(*color, PN.trail_alpha[0] + PN.trail_alpha[1] * i / len(pts)), lw=PN.trail_width * sc)
        ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=(*color, 1.0), lw=PN.arrow["lw"] * sc, mutation_scale=PN.arrow["mutation"] * sc))
        sgn = angular_velocity_sign(helicity, view)
        RA = PN.rotation_arc
        arc = np.linspace(RA["start"], RA["end"], RA["points"]) * sgn
        ax.plot(RA["radius"] * np.cos(arc), RA["radius"] * np.sin(arc), color=(1, 1, 1, RA["alpha"]), lw=RA["width"] * sc)
        ax.annotate("", xy=(RA["radius"] * math.cos(RA["head_to"] * sgn), RA["radius"] * math.sin(RA["head_to"] * sgn)),
                    xytext=(RA["radius"] * math.cos(RA["head_from"] * sgn), RA["radius"] * math.sin(RA["head_from"] * sgn)),
                    arrowprops=dict(arrowstyle="-|>", color=(1, 1, 1, RA["head_alpha"]), lw=RA["width"] * sc, mutation_scale=RA["head_mutation"] * sc))
        ax.set_title(title, color=TXT, fontsize=PN.title_size * sc, pad=PN.title_pad)
        ax.set_xlabel(tx["ccw"] if sgn > 0 else tx["cw"], color=(*color, 1.0), fontsize=PN.label_size * sc, labelpad=PN.label_pad)

    for k_ in ids:
        t_film = k_ / fps
        t = t_film / k
        fig.texts.clear()
        fig.patches.clear()
        for art in list(fig.artists):
            art.remove()
        scene = 1 if t < TLN.scene_ends[0] else 2 if t < TLN.scene_ends[1] else 3 if t < TLN.scene_ends[2] else 4
        for a_ in (d3, v1, v2, big, *spin_axes):
            a_.clear()
            a_.axis("off")
            a_.set_visible(False)

        if scene in (1, 2):
            hel = +1 if scene == 1 else -1
            col = C_POS if hel > 0 else C_NEG
            d3.set_visible(True)
            d3.set_xlim(*S3.xlim)
            d3.set_ylim(*S3.ylim)
            d3.set_aspect("equal")
            zs = np.linspace(0, L_Z, S3.samples)
            e = field_circular(t, zs, hel)
            amp = S3.amplitude
            AA = S3.axis_arrow
            Xz, Yz = proj(0, 0, L_Z + AA["extra"])
            d3.annotate("", xy=(Xz, Yz), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=(*ST.axis_arrow, AA["alpha"]), lw=AA["lw"] * sc, mutation_scale=AA["mutation"] * sc), zorder=2)
            d3.text(Xz + S3.z_label["dx"], Yz + S3.z_label["dy"], tx["axis_z"], color=TXT, fontsize=S3.z_label["size"] * sc)
            xe, ye = amp * e[:, 0], amp * e[:, 1]
            Xs, Ys = proj(xe, ye, zs)
            Xa, Ya = proj(0 * zs, 0 * zs, zs)
            # depth coordinate towards the viewer: larger = nearer
            depth = (zs * math.sin(phi_c) + ye * math.cos(phi_c)) * math.cos(th_c) - xe * math.sin(th_c)
            near = 0.5 + 0.5 * np.tanh(-depth / S3.depth_scale)
            segs, cols, lws = [], [], []
            for i in range(len(zs) - 1):
                segs.append(np.array([[Xs[i], Ys[i]], [Xs[i + 1], Ys[i + 1]]]))
                cols.append((*col, S3.tip_alpha[0] + S3.tip_alpha[1] * near[i]))
                lws.append((S3.tip_width[0] + S3.tip_width[1] * near[i]) * sc)
            for i in range(0, len(zs), S3.stem_step):
                segs.append(np.array([[Xa[i], Ya[i]], [Xs[i], Ys[i]]]))
                cols.append((*col, S3.stem_alpha[0] + S3.stem_alpha[1] * near[i]))
                lws.append(S3.stem_width * sc)
            d3.add_collection(LineCollection(segs, colors=cols, linewidths=lws, capstyle="round", zorder=3))
            # the transverse axes at the plane z = z_obs and the E vector there
            zo = S3.observer_z
            eo = field_circular(t, zo, hel)
            Xo, Yo = proj(amp * eo[0], amp * eo[1], zo)
            Xa, Ya = proj(0, 0, zo)
            OA = S3.observer_arrow
            d3.annotate("", xy=(Xo, Yo), xytext=(Xa, Ya), arrowprops=dict(arrowstyle="-|>", color=(1, 1, 1, OA["alpha"]), lw=OA["lw"] * sc, mutation_scale=OA["mutation"] * sc), zorder=7)
            # the observer's plane
            OR = S3.observer_ring
            ph = np.linspace(0, 2 * math.pi, OR["points"])
            Xc, Yc = proj(amp * np.cos(ph), amp * np.sin(ph), np.full_like(ph, zo))
            d3.plot(Xc, Yc, color=(1, 1, 1, OR["alpha"]), lw=OR["width"] * sc, ls=(0, tuple(OR["dash"])))
            d3.set_visible(True)
            view_panel(v1, hel, "receiver", t, col, tx["recv"])
            view_panel(v2, hel, "source", t, col, tx["src"])
            v1.set_visible(True)
            v2.set_visible(True)
            fig.text(PT.x, PT.name["y"], tx["pos"] if hel > 0 else tx["neg"], color=(*col, 1.0), fontsize=PT.name["size"] * sc)
            fig.text(PT.x, PT.names["y"], tx["names_pos"] if hel > 0 else tx["names_neg"], color=DIM, fontsize=PT.names["size"] * sc)
            fig.text(PT.x, PT.convention["y"], tx["conv"], color=DIM, fontsize=PT.convention["size"] * sc, va="top", linespacing=PT.convention["linespacing"])
            fig.text(PT.x, PT.formula_eps["y"], tx["fP"] if hel > 0 else tx["fN"], color=TXT, fontsize=PT.formula_eps["size"] * sc)
            fig.text(PT.x, PT.formula_e["y"], tx["fEP"] if hel > 0 else tx["fEN"], color=TXT, fontsize=PT.formula_e["size"] * sc)
            fig.text(PT.snap["x"], PT.snap["y"], tx["snap"] if hel > 0 else tx["snap_neg"], color=(*col, 1.0), fontsize=PT.snap["size"] * sc)
            fig.text(PT.time["x"], PT.time["y"], tx["time"] if hel > 0 else tx["time_neg"], color=DIM, fontsize=PT.time["size"] * sc)
            cap = tx["s1"] if hel > 0 else tx["s2"]
        elif scene == 3:
            big.set_visible(True)
            big.set_xlim(-MX.lim, MX.lim)
            big.set_ylim(-MX.lim, MX.lim)
            big.set_aspect("equal")
            a_l, a_r, delta = mix_parameters(t)
            a = np.linspace(0, 2 * math.pi, MX.circle_points)
            big.plot(np.cos(a), np.sin(a), color=(*EDGE, MX.circle_alpha), lw=PN.circle_width * sc)
            big.plot([-MX.cross_half, MX.cross_half], [0, 0], color=(1, 1, 1, MX.cross_alpha), lw=PN.cross_width * sc)
            big.plot([0, 0], [-MX.cross_half, MX.cross_half], color=(1, 1, 1, MX.cross_alpha), lw=PN.cross_width * sc)
            # the polarization ellipse (the path of the sum)
            tt = np.linspace(0, 2 * math.pi / OMEGA, MX.ellipse_samples)
            pts = np.array([field_from_amplitudes(x_, a_l, a_r, delta)[:2] for x_ in tt])
            big.plot(pts[:, 0], pts[:, 1], color=(1, 1, 1, MX.ellipse_alpha), lw=MX.ellipse_width * sc, ls=(0, tuple(MX.ellipse_dash)))
            eL = a_l * np.array([math.cos(OMEGA * t), math.sin(OMEGA * t)])
            eR = a_r * np.array([math.cos(OMEGA * t + delta), -math.sin(OMEGA * t + delta)])
            eS = eL + eR
            VC = MX.vector
            # the two rotating vectors, placed head to tail
            big.annotate("", xy=tuple(eL), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=(*C_POS, VC["alpha"]), lw=VC["lw"] * sc, mutation_scale=VC["mutation"] * sc))
            big.annotate("", xy=tuple(eS), xytext=tuple(eL), arrowprops=dict(arrowstyle="-|>", color=(*C_NEG, VC["alpha"]), lw=VC["lw"] * sc, mutation_scale=VC["mutation"] * sc))
            big.annotate("", xy=tuple(eS), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=(*C_SUM, 1.0), lw=VC["sum_lw"] * sc, mutation_scale=VC["sum_mutation"] * sc))
            trail = np.array([field_from_amplitudes(x_, a_l, a_r, delta)[:2] for x_ in np.linspace(t - MX.trail_seconds, t, MX.trail_points)])
            for i in range(len(trail) - 1):
                big.plot(trail[i:i + 2, 0], trail[i:i + 2, 1], color=(1, 1, 1, MX.trail_alpha[0] + MX.trail_alpha[1] * i / len(trail)), lw=MX.trail_width * sc)
            big.text(MX.e_label["x"], MX.e_label["y"], tx["e_label"], color=TXT, fontsize=MX.e_label["size"] * sc, ha="right")
            big.set_visible(True)
            major, minor = ellipse_axes(a_l, a_r)
            ratio = minor / major
            kind = tx["linear"] if ratio < MX.linear_ratio else tx["circle"] if ratio > MX.circle_ratio or a_r < MX.circle_ar else tx["ellipse"]
            fig.text(RD.x, RD.kind["y"], kind, color=TXT, fontsize=RD.kind["size"] * sc)
            fig.text(RD.x, RD.formula["y"], tx["fmix"], color=TXT, fontsize=RD.formula["size"] * sc)
            fig.text(RD.x, RD.ar["y"], rf"$a_R={num(a_l, '.2f', lang)}$", color=C_POS, fontsize=RD.ar["size"] * sc)
            fig.text(RD.x, RD.al["y"], rf"$a_L={num(a_r, '.2f', lang)}$", color=C_NEG, fontsize=RD.al["size"] * sc)
            fig.text(RD.x, RD.delta["y"], rf"$\delta={math.degrees(delta):.0f}^\circ$", color=DIM, fontsize=RD.delta["size"] * sc)
            fig.text(RD.x, RD.axes_title["y"], tx["axes"], color=DIM, fontsize=RD.axes_title["size"] * sc)
            fig.text(RD.x, RD.axes_value["y"], rf"$a_R+a_L={num(major, '.2f', lang)},\quad |a_R-a_L|={num(minor, '.2f', lang)}$", color=TXT, fontsize=RD.axes_value["size"] * sc)
            tilt_deg = math.degrees(ellipse_tilt(delta)) + 0.0
            tilt_deg = 0.0 if abs(tilt_deg) < MX.tilt_zero else tilt_deg
            fig.text(RD.x, RD.tilt_title["y"], tx["tilt"], color=DIM, fontsize=RD.tilt_title["size"] * sc)
            fig.text(RD.x, RD.tilt_value["y"], rf"$-\delta/2={tilt_deg:.0f}^\circ$", color=TXT, fontsize=RD.tilt_value["size"] * sc)
            fig.text(RD.legend_x[0], RD.legend_y, tx["leg_R"], color=C_POS, fontsize=RD.legend_size * sc)
            fig.text(RD.legend_x[1], RD.legend_y, tx["leg_L"], color=C_NEG, fontsize=RD.legend_size * sc)
            fig.text(RD.legend_x[2], RD.legend_y, tx["leg_sum"], color=C_SUM, fontsize=RD.legend_size * sc)
            cap = tx["s3a"] if t < TLN.caption_steps[0] else tx["s3b"] if t < TLN.caption_steps[1] else tx["s3c"]
        else:
            # the photon spin
            big.set_visible(True)
            big.set_xlim(-MX.lim, MX.lim)
            big.set_ylim(-MX.lim, MX.lim)
            big.set_aspect("equal")
            for i, (hel, col) in enumerate(((+1, C_POS), (-1, C_NEG))):
                ax_ = spin_axes[i]
                ax_.set_visible(True)
                view_panel(ax_, hel, "receiver", t, col, tx["pos"] if hel > 0 else tx["neg"])
                fig.text(SA["x0"] + SA["dx"] * i + RD.spin_value["dx"], RD.spin_value["y"], tx["sz_pos"] if hel > 0 else tx["sz_neg"], color=col, fontsize=RD.spin_value["size"] * sc, ha="center")
            fig.text(RD.spin_label["x"], RD.spin_label["y"], tx["spin"], color=TXT, fontsize=RD.spin_label["size"] * sc)
            cap = tx["s4"]
        fig.text(*LY.title_pos, tx["title"], color=tuple(ST.title_color), fontsize=LY.title_size * sc)
        fig.text(*LY.subtitle_pos, tx["sub"], color=DIM, fontsize=LY.subtitle_size * sc)
        fig.text(*LY.caption_pos, cap, color=TXT, fontsize=LY.caption_size * sc)
        fade_io = min(smooth(t_film, 0.0, V.fade_s), 1.0 - smooth(t_film, total - V.fade_s, total))
        for edge in TLN.scene_ends:
            if abs(t - edge) < TLN.scene_fade_s:
                fade_io = min(fade_io, TLN.scene_fade_floor + (1.0 - TLN.scene_fade_floor) * abs(t - edge) / TLN.scene_fade_s)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)
        if fade_io < 1.0:
            frame = bg_rgba + (frame - bg_rgba) * fade_io
        frame = frame.clip(0, 255).astype(np.uint8)
        if writer is None:
            from PIL import Image
            Image.fromarray(frame).save(out)
        else:
            writer.stdin.write(frame.tobytes())
    if writer is not None:
        writer.stdin.close()
        writer.wait()
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=("en", "ru"), default="en")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=TOTAL)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"circular_polarization_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
