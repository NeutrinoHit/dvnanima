r"""Twisted (vortex) beams: a phase that winds m times around the axis carries the orbital angular momentum L_z = m hbar.

The film follows the book (the free Schroedinger / Helmholtz equation, (nabla^2 + k^2) psi = 0):

  * A Bessel (cylindrical) wave  psi = J_m(kappa rho) exp[i (m phi + k_z z - omega t)]  has the definite
    projection L_z = -i hbar d/dphi = m hbar.  Its phase winds m times around the axis, so on the axis (m != 0)
    the phase is undefined and the intensity J_m^2 vanishes: a vortex with a dark core.
  * Constant-phase surfaces, m phi + k_z z = omega t + 2 pi n, are m intertwined helicoids (a screw with m
    threads): one turn of the wave front per wavelength is replaced by m turns around the axis.
  * A Gaussian beam passes a spiral phase plate (or a spatial light modulator with the pattern exp(i m phi)):
    the field becomes the Laguerre-Gauss mode LG_0^m ~ (sqrt2 rho / w)^|m| exp(-rho^2/w^2) exp(i m phi), a ring of
    radius  w sqrt(|m|/2)  with the same winding of the phase.  The sign of m gives the handedness.

Colour = phase of psi (the colour wheel), brightness = |psi|.

Film time (s), 56 s:  0-15 transverse phase portraits for m = 0, 1, 2, 3; 15-33 the helicoidal wave fronts in 3D
(m = 1, 2, 3, -2); 33-50 the Gaussian beam and the spiral phase plate (m = 1, 2, 3); 50-56 summary.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python oam_beams.py --lang en            # film -> media/oam_beams_en.mp4
    python oam_beams.py --lang ru
    python oam_beams.py --lang en --snapshot 20
"""

from __future__ import annotations

import argparse
import colorsys
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import jv, jnp_zeros

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config, load_texts  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)

TOTAL = CFG.timeline.total
KAPPA = CFG.beam.kappa                  # transverse wave number of the Bessel beams, in 1 / (length unit)
KZ = CFG.beam.kz                        # longitudinal wave number of the 3D wave fronts
OMEGA = 2 * math.pi * CFG.beam.omega_cycles    # the film frequency of the phase rotation, rad per unit time
W_GAUSS = CFG.beam.w_gauss              # waist of the Gaussian beam


# ----------------------------------------------------------------------------- fields

def polar(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.hypot(x, y), np.arctan2(y, x)


def bessel_beam(x: np.ndarray, y: np.ndarray, m: int, kappa: float = KAPPA) -> np.ndarray:
    """Transverse profile J_m(kappa rho) exp(i m phi) of a Bessel (cylindrical) wave."""
    rho, phi = polar(x, y)
    return jv(m, kappa * rho) * np.exp(1j * m * phi)


def lg_beam(x: np.ndarray, y: np.ndarray, m: int, w: float = W_GAUSS) -> np.ndarray:
    """Laguerre-Gauss mode LG_0^m at the waist: (sqrt2 rho / w)^|m| exp(-rho^2/w^2) exp(i m phi)."""
    rho, phi = polar(x, y)
    return (np.sqrt(2) * rho / w) ** abs(m) * np.exp(-rho ** 2 / w ** 2) * np.exp(1j * m * phi)


def gauss_beam(x: np.ndarray, y: np.ndarray, w: float = W_GAUSS) -> np.ndarray:
    rho, _ = polar(x, y)
    return np.exp(-rho ** 2 / w ** 2) + 0j


def spiral_plate(x: np.ndarray, y: np.ndarray, m: int) -> np.ndarray:
    """Transmission exp(i m phi) of a spiral phase plate / a spatial light modulator."""
    _, phi = polar(x, y)
    return np.exp(1j * m * phi)


def lg_ring_radius(m: int, w: float = W_GAUSS) -> float:
    return w * math.sqrt(abs(m) / 2.0)


def bessel_ring_radius(m: int, kappa: float = KAPPA) -> float:
    """Radius of the first intensity maximum of J_m^2(kappa rho): the first zero of J_m'."""
    if m == 0:
        return 0.0
    return float(jnp_zeros(abs(m), 1)[0]) / kappa


def winding_number(psi_loop: np.ndarray) -> int:
    """Number of times the phase of psi winds around a closed loop (given as a sampled array)."""
    ph = np.unwrap(np.angle(np.concatenate([psi_loop, psi_loop[:1]])))
    return int(round((ph[-1] - ph[0]) / (2 * math.pi)))


def lz_expectation(psi: np.ndarray, phi_axis: np.ndarray) -> float:
    """<-i d/dphi> of psi(rho, phi) sampled on a uniform phi grid (periodic), weighted by |psi|^2."""
    dphi = phi_axis[1] - phi_axis[0]
    d = (np.roll(psi, -1, axis=-1) - np.roll(psi, 1, axis=-1)) / (2 * dphi)
    num = np.sum(np.conj(psi) * (-1j) * d)
    den = np.sum(np.abs(psi) ** 2)
    return float((num / den).real)


def wavefront_phi(m: int, n: int, z: np.ndarray, t: float, kz: float = KZ, omega: float = OMEGA) -> np.ndarray:
    """Azimuth of the n-th helicoid m phi + kz z - omega t = 2 pi n at the heights z."""
    return (2 * math.pi * n + omega * t - kz * z) / m


def phase_colors(psi: np.ndarray, gamma: float = CFG.phase_image.gamma, vmax: float | None = None) -> np.ndarray:
    """RGB image: hue = phase, value = |psi| (normalised)."""
    amp = np.abs(psi)
    amp = amp / (vmax if vmax else amp.max() or 1.0)
    h = (np.angle(psi) / (2 * math.pi)) % 1.0
    v = np.clip(amp, 0, 1) ** gamma
    s = np.full_like(h, CFG.phase_image.saturation)
    hsv = np.stack([h, s, v], axis=-1)
    from matplotlib.colors import hsv_to_rgb
    return hsv_to_rgb(hsv)


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.lines
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, GR, TLN, PI_, HC, IN, PNL, LY, ST = (CFG.video, CFG.grid, CFG.timeline, CFG.phase_image, CFG.helicoid, CFG.inset, CFG.panels, CFG.layout, CFG.style)
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = ST.background
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)
    TXT = tuple(ST.text)
    DIM = tuple(ST.dim)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    N = GR.points_full if W >= GR.wide_width else GR.points_preview
    EXT = GR.extent
    gx = np.linspace(-EXT, EXT, N)
    X, Y = np.meshgrid(gx, gx)
    imgs = [fig.add_axes([PNL.x0 + PNL.dx * i, PNL.y, PNL.width, PNL.height], facecolor="none") for i in range(4)]
    d3 = fig.add_axes(HC.axes, facecolor="none")
    inset = fig.add_axes(IN.axes, facecolor="none")
    k = total / TOTAL
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    az, el = math.radians(HC.view["azimuth_deg"]), math.radians(HC.view["elevation_deg"])

    def pcx(i: int) -> float:
        return PNL.x0 + PNL.dx * i + PNL.centre_dx

    def proj(x, y, z):
        x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
        X_ = x * math.cos(az) - y * math.sin(az)
        Y_ = z * math.cos(el) + (x * math.sin(az) + y * math.cos(az)) * math.sin(el)
        return X_, Y_

    def show_img(ax, rgb, title=None, sub=None, ring=None):
        ax.clear()
        ax.set_facecolor("none")
        ax.imshow(rgb, extent=(-EXT, EXT, -EXT, EXT), origin="lower", interpolation="bilinear")
        ax.set_xlim(-EXT, EXT)
        ax.set_ylim(-EXT, EXT)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(tuple(PNL.border))
            s.set_linewidth(PNL.border_width * sc)
        if ring:
            ax.add_patch(mp.Circle((0, 0), ring, fill=False, ec=(1, 1, 1, PNL.ring_alpha), lw=PNL.ring_width * sc, ls=(0, tuple(PNL.ring_dash))))
        if title:
            ax.set_title(title, color=TXT, fontsize=PNL.title_size * sc, pad=PNL.title_pad)
        if sub:
            ax.set_xlabel(sub, color=DIM, fontsize=PNL.sub_size * sc, labelpad=4)

    for k_ in ids:
        t_film = k_ / fps
        t = t_film / k
        fig.texts.clear()
        fig.patches.clear()
        for art in list(fig.artists):
            art.remove()
        phase_t = OMEGA * t
        scene = 1 if t < TLN.scene_ends[0] else 2 if t < TLN.scene_ends[1] else 3 if t < TLN.scene_ends[2] else 4

        for a in imgs + [d3, inset]:
            a.clear()
            a.axis("off")
            a.set_visible(False)

        if scene == 1:
            for i, m in enumerate((0, 1, 2, 3)):
                psi = bessel_beam(X, Y, m) * np.exp(-1j * phase_t)
                rgb = phase_colors(psi, vmax=PI_.vmax_plane if m == 0 else PI_.vmax_vortex)
                imgs[i].set_visible(True)
                show_img(imgs[i], rgb, title=rf"$m={m}$", sub=tx["bright"] if m == 0 else tx["dark"])
                # a loop around the axis with an arrow showing the winding
                if m > 0:
                    a_ = np.linspace(0, 2 * math.pi, PNL.loop_points)
                    imgs[i].plot(PNL.loop_radius * np.cos(a_), PNL.loop_radius * np.sin(a_), color=(1, 1, 1, PNL.loop_alpha), lw=PNL.loop_width * sc, ls=(0, tuple(PNL.loop_dash)))
                    fig.text(pcx(i), PNL.label_y, tx["lz"].format(m=m), color=TXT, fontsize=PNL.label_size * sc, ha="center")
                else:
                    fig.text(pcx(i), PNL.label_y, tx["lz"].format(m=0), color=TXT, fontsize=PNL.label_size * sc, ha="center")
            fig.text(*LY.formula1_pos, tx["f1"], color=TXT, fontsize=LY.formula_size * sc)
            fig.text(*LY.formula2_pos, tx["f2"], color=TXT, fontsize=LY.formula_size * sc)
            cap = tx["s1"]
        elif scene == 2:
            d3.set_visible(True)
            inset.set_visible(True)
            d3.set_xlim(*HC.xlim)
            d3.set_ylim(*HC.ylim)
            d3.set_aspect("equal")
            c1, c2, c3 = TLN.scene2_m_changes
            if t < c1:
                m = 1
            elif t < c2:
                m = 2
            elif t < c3:
                m = 3
            else:
                m = -2
            mm = abs(m)
            z0 = HC.z0
            R = HC.radius
            r_in = HC.core_fraction * R          # the dark core is left empty
            L_Z = HC.length
            nz, nr = HC.z_samples, HC.r_samples
            zs = np.linspace(0.0, L_Z, nz + 1)
            rs = np.linspace(r_in, R, nr + 1)
            quads, qcols, depth = [], [], []
            for n in range(mm):
                hue = (n / mm + HC.hue_offset) % 1.0
                rgb = colorsys.hsv_to_rgb(hue, HC.saturation, HC.value)
                phis = (2 * math.pi * n + OMEGA * t - KZ * zs) / m
                for iz in range(nz):
                    for ir in range(nr):
                        r0, r1 = rs[ir], rs[ir + 1]
                        p0, p1 = phis[iz], phis[iz + 1]
                        xs = np.array([r0 * math.cos(p0), r1 * math.cos(p0), r1 * math.cos(p1), r0 * math.cos(p1)])
                        ys = np.array([r0 * math.sin(p0), r1 * math.sin(p0), r1 * math.sin(p1), r0 * math.sin(p1)])
                        zq = np.array([zs[iz], zs[iz], zs[iz + 1], zs[iz + 1]]) + z0
                        X_, Y_ = proj(xs, ys, zq)
                        quads.append(np.column_stack([X_, Y_]))
                        shade = HC.shade_base + HC.shade_gain * (0.5 + 0.5 * math.cos(0.5 * (p0 + p1) - az))
                        zmid = 0.5 * (zs[iz] + zs[iz + 1])
                        edge = math.sin(math.pi * zmid / L_Z) ** HC.edge_exponent
                        qcols.append((rgb[0] * shade, rgb[1] * shade, rgb[2] * shade, HC.alpha_base + HC.alpha_gain * edge))
                        depth.append(0.5 * (xs.mean() * math.sin(az) + ys.mean() * math.cos(az)))
            order = np.argsort(depth)
            pc = PolyCollection([quads[i] for i in order], facecolors=[qcols[i] for i in order],
                                edgecolors=[tuple(HC.edge_colour)] * len(order), linewidths=HC.edge_width * sc, zorder=3)
            d3.add_collection(pc)
            # the bright ring of the beam at the end plane
            # the axis and the end ring
            Xa, Ya = proj([0, 0], [0, 0], [z0 - HC.axis_margin[0], z0 + HC.axis_margin[1]])
            d3.plot(Xa, Ya, color=(*ST.axis_colour, HC.axis_alpha), lw=HC.axis_width * sc, ls=(0, tuple(HC.axis_dash)), zorder=2)
            d3.annotate("", xy=(Xa[1], Ya[1]), xytext=(Xa[1] - HC.arrow_head, Ya[1] - HC.arrow_head), arrowprops=dict(arrowstyle="-|>", color=(*ST.axis_colour, HC.arrow_alpha), lw=HC.arrow_width * sc))
            d3.text(Xa[1] + HC.z_label["dx"], Ya[1] + HC.z_label["dy"], r"$z$", color=TXT, fontsize=HC.z_label["size"] * sc)
            ph = np.linspace(0, 2 * math.pi, HC.ring_points)
            for zr in (z0, z0 + L_Z):
                Xc, Yc = proj(R * np.cos(ph), R * np.sin(ph), np.full_like(ph, zr))
                d3.plot(Xc, Yc, color=(*ST.ring_colour, HC.ring_alpha), lw=HC.ring_width * sc, zorder=2)
            # the dark core
            Xc, Yc = proj(np.zeros(2), np.zeros(2), np.array([z0, z0 + L_Z]))
            d3.plot(Xc, Yc, color=(0, 0, 0, 0.0))
            # the transverse pattern of the beam
            psi = bessel_beam(X, Y, m) * np.exp(-1j * phase_t)
            rgb = phase_colors(psi, vmax=PI_.vmax_vortex if m else PI_.vmax_plane)
            show_img(inset, rgb, title=rf"$m={m}$".replace("-", "\u2212"), sub=tx["right"] if m > 0 else tx["left"])
            inset.set_visible(True)
            fig.text(*IN.lz_pos, tx["lz"].format(m=str(m).replace("-", "−")), color=TXT, fontsize=IN.lz_size * sc)
            fig.text(*IN.f3_pos, tx["f3"], color=TXT, fontsize=IN.f3_size * sc)
            cap = tx["s2"]
        elif scene == 3:
            lt = t - TLN.scene3_start
            m = 1 if lt < TLN.scene3_m_changes[0] else 2 if lt < TLN.scene3_m_changes[1] else 3
            titles = (tx["gauss"], tx["plate"], tx["after"], tx["after"])
            psi_g = gauss_beam(X, Y)
            plate = spiral_plate(X, Y, m)
            psi_after = psi_g * plate
            # the near field after the plate is the Gaussian with the spiral phase; the LG ring forms on propagation
            ring_field = lg_beam(X, Y, m)
            ring_field = ring_field / np.abs(ring_field).max()
            rgb0 = phase_colors(psi_g * np.exp(-1j * phase_t), vmax=1.0)
            rgb1 = phase_colors(np.exp(1j * np.angle(plate)) * (np.abs(psi_g) > -1) * 1.0, vmax=1.0, gamma=1.0)
            rgb2 = phase_colors(psi_after * np.exp(-1j * phase_t), vmax=1.0)
            rgb3 = phase_colors(ring_field * np.exp(-1j * phase_t), vmax=1.0)
            for i, (rgb, tt) in enumerate(zip((rgb0, rgb1, rgb2, rgb3), (tx["gauss"], tx["plate"], tx["after"], tx["ring"] + (rf" ${lg_ring_radius(m):.2f}$".replace(".", "{,}") if lang == "ru" else rf" ${lg_ring_radius(m):.2f}$")))):
                imgs[i].set_visible(True)
                show_img(imgs[i], rgb, title=None)
                fig.text(pcx(i), PNL.scene3_title_y, tt, color=TXT, fontsize=PNL.scene3_title_size * sc, ha="center")
            show_img(imgs[3], rgb3, ring=lg_ring_radius(m))
            fig.text(pcx(3), PNL.label_y, tx["f5"], color=TXT, fontsize=PNL.f5_size * sc, ha="center")
            AR = PNL.arrow
            for i in range(3):
                x0 = PNL.x0 + PNL.dx * i + PNL.width + AR["pad"]
                fig.add_artist(matplotlib.patches.FancyArrowPatch((x0 + AR["start"], AR["y"]), (x0 + AR["end"], AR["y"]), transform=fig.transFigure,
                                                                  arrowstyle="-|>", mutation_scale=AR["mutation"] * sc, color=(1, 1, 1, AR["alpha"]), lw=AR["lw"] * sc))
            for i in range(4):
                fig.text(pcx(i), PNL.label_y, ["", "", tx["lz"].format(m=m), ""][i], color=TXT, fontsize=PNL.label_size * sc, ha="center")
            fig.text(*LY.formula1_pos, tx["f4"], color=TXT, fontsize=LY.formula_size * sc)
            fig.text(pcx(1), PNL.m_label["y"], rf"$m={m}$", color=tuple(PNL.m_label["colour"]), fontsize=PNL.m_label["size"] * sc, ha="center")
            cap = tx["s3"]
        else:
            # summary: m = +2 and m = -2 side by side with the ring and the winding
            for i, m in enumerate((2, -2)):
                psi = lg_beam(X, Y, m) * np.exp(-1j * phase_t)
                psi = psi / np.abs(psi).max()
                imgs[i + 1].set_visible(True)
                show_img(imgs[i + 1], phase_colors(psi, vmax=1.0), title=rf"$m={m:+d}$".replace("-", "\u2212"), sub=tx["right"] if m > 0 else tx["left"])
                fig.text(pcx(i + 1), PNL.label_y, tx["lz"].format(m=f"{m:+d}".replace("-", "−")), color=TXT, fontsize=PNL.label_size * sc, ha="center")
            fig.text(*LY.formula1_pos, tx["f2"], color=TXT, fontsize=LY.formula_size * sc)
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
    out = args.out or HERE / "media" / f"oam_beams_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
