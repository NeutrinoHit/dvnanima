r"""How a photomultiplier tube amplifies a single photon.

Photon -> photocathode (photoelectron) -> a chain of N dynodes, each at a higher potential than the
previous one (about 100 V per stage) -> anode.  An electron hitting a dynode knocks out a random number
of secondary electrons with mean delta (Poisson), so the number of electrons grows like a branching
process:

    n_0 = number of photoelectrons,     n_j = sum_{i=1}^{n_{j-1}} Poisson(delta),     <n_j> = n_0 delta^j.

The gain of the tube is G = n_N / n_0 ~ delta^N (here 4^10 ~ 10^6).  Because the amplification is
linear, the anode pulse is proportional to the number of photoelectrons, i.e. to the intensity of the
light flash (for a scintillator, to the energy deposited).

Scene 1 (0-20 s).  One photoelectron, stage by stage.  Bars: the random electron counts n_j against the
average n_0 delta^j.
Scene 2 (20-40 s).  Three scintillation flashes of 2, 6 and 12 photoelectrons: three anode pulses
whose heights are in the ratio of the photoelectron numbers.

Schematic: the electron paths are drawn as arcs, only up to 36 electrons per stage are shown (the
counter shows all of them), the stage time is the same for every electron up to a small jitter.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python pmt_multiplication.py --lang en       # film -> media/pmt_multiplication_en.mp4
    python pmt_multiplication.py --lang ru
    python pmt_multiplication.py --lang en --snapshot 12
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config, load_texts  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)
_T = CFG.tube

DELTA = CFG.physics.delta               # mean secondary-emission yield per dynode
N_DYN = CFG.physics.dynodes             # number of dynodes
MAX_DRAWN = CFG.physics.max_drawn       # electrons drawn per stage
CATHODE_X = _T.cathode_x
ANODE_X = _T.anode_x
DYN_X0, DYN_DX = _T.dynode_x0, _T.dynode_dx
Y_TOP, Y_BOT, Y_MID = _T.y_top, _T.y_bottom, _T.y_mid
PLATE_HALF = _T.plate_half
EVENTS_1 = tuple(tuple(e) for e in CFG.events.scene1)                 # (start time s, photoelectrons) in scene 1
EVENTS_2 = tuple(tuple(e) for e in CFG.events.scene2)                 # scene 2
SEEDS = {int(k): v for k, v in CFG.events.seeds.to_dict().items()}


# ------------------------------------------------------------------ physics

def multiply(n0: int, rng: np.random.Generator, delta: float = DELTA, n_dyn: int = N_DYN) -> list[int]:
    """Random electron numbers [n_0, n_1, ..., n_N]: a Galton-Watson process with Poisson(delta) offspring."""
    counts = [int(n0)]
    for _ in range(n_dyn):
        counts.append(int(rng.poisson(delta * counts[-1])) if counts[-1] > 0 else 0)
    return counts


def mean_counts(n0: int, delta: float = DELTA, n_dyn: int = N_DYN) -> list[float]:
    return [n0 * delta ** j for j in range(n_dyn + 1)]


def dynode_center(j: int) -> tuple[float, float]:
    """Centre of dynode j = 1..N (alternately top and bottom row)."""
    return DYN_X0 + DYN_DX * (j - 1), (Y_TOP if j % 2 == 1 else Y_BOT)


def plate_endpoints(j: int) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = dynode_center(j)
    a = math.radians(_T.plate_angle_deg) * (1 if j % 2 == 1 else -1)
    d = np.array([math.cos(a), math.sin(a)]) * PLATE_HALF
    return np.array([cx, cy]) - d, np.array([cx, cy]) + d


def node_point(j: int, rng: np.random.Generator) -> np.ndarray:
    """Random emission/landing point of stage j: 0 photocathode, 1..N dynode, N+1 anode."""
    if j == 0:
        return np.array([CATHODE_X + _T.cathode_depth * rng.random(), Y_MID + _T.cathode_spread * (rng.random() - 0.5) * 2 * _T.spread_factor])
    if j == N_DYN + 1:
        return np.array([ANODE_X, Y_MID + _T.anode_spread * (rng.random() - 0.5) * 2 * _T.spread_factor])
    p0, p1 = plate_endpoints(j)
    s = _T.landing_range[0] + _T.landing_range[1] * rng.random()
    inward = np.array([0.0, -_T.inward if j % 2 == 1 else _T.inward])
    return p0 + s * (p1 - p0) + inward


def bezier(a: np.ndarray, b: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    s = s[..., None]
    return (1 - s) ** 2 * a + 2 * (1 - s) * s * c + s ** 2 * b


@dataclass
class Flight:
    """The drawn electrons of one stage: from node j to node j + 1."""
    a: np.ndarray        # (m, 2) start
    b: np.ndarray        # (m, 2) end
    c: np.ndarray        # (m, 2) bezier control point
    depart: np.ndarray   # (m,)
    arrive: np.ndarray   # (m,)


@dataclass
class Event:
    t0: float
    n_pe: int
    counts: list[int]
    stage_dt: float
    flights: list[Flight]

    def stage_time(self, j: int) -> float:
        """Nominal time at which the electrons of stage j leave node j."""
        return self.t0 + _T.start_delay + j * self.stage_dt

    @property
    def anode_time(self) -> float:
        return self.stage_time(N_DYN + 1) - _T.anode_lead * self.stage_dt


def make_event(t0: float, n_pe: int, stage_dt: float) -> Event:
    rng = np.random.default_rng(SEEDS.get(n_pe, CFG.events.default_seed))
    counts = multiply(n_pe, rng)
    flights = []
    for j in range(N_DYN + 1):
        m = min(counts[j], MAX_DRAWN)
        a = np.array([node_point(j, rng) for _ in range(m)]).reshape(m, 2)
        b = np.array([node_point(j + 1, rng) for _ in range(m)]).reshape(m, 2)
        mid = 0.5 * (a + b)
        c = mid + np.column_stack([_T.control_jitter_x * (rng.random(m) - 0.5), (Y_MID - mid[:, 1]) * _T.control_pull + _T.control_jitter_y * (rng.random(m) - 0.5)])
        t_dep = t0 + _T.start_delay + j * stage_dt + stage_dt * _T.depart_jitter * rng.random(m)
        t_arr = t_dep + stage_dt * (_T.arrive_range[0] + _T.arrive_range[1] * rng.random(m))
        flights.append(Flight(a, b, c, t_dep, t_arr))
    return Event(t0, n_pe, counts, stage_dt, flights)


def pulse_shape(t: np.ndarray, t_c: float, height: float, width: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((t - t_c) / width) ** 2)


def smooth(v: float, a: float, b: float) -> float:
    u = min(max((v - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


# --------------------------------------------------------------------- film

TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def num(x: float, fmt: str, lang: str) -> str:
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def sci(x: float, lang: str) -> str:
    """x as a LaTeX number a \\times 10^b (inside $...$)."""
    mant, exp = f"{x:.2g}".replace("e+0", "e").replace("e+", "e").split("e") if "e" in f"{x:.2g}" else (f"{x:.2g}", "0")
    if exp == "0":
        return mant.replace(".", "{,}") if lang == "ru" else mant
    return (mant.replace(".", "{,}") if lang == "ru" else mant) + rf"\times10^{{{int(exp)}}}"


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str = "en", snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, ST, LY, TA, BP, PP, TL = CFG.video, CFG.style, CFG.layout, CFG.tube_axes, CFG.bars_panel, CFG.pulse_panel, CFG.timeline
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = ST.background
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)
    TXT = tuple(ST.text)
    DIM = tuple(ST.dim)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax_t = fig.add_axes(TA.axes, facecolor="none")
    ax_b = fig.add_axes(BP.axes, facecolor="none")
    ax_p = fig.add_axes(PP.axes, facecolor="none")

    k = total / TL.film_length
    scene1 = [make_event(t0 * k, n, CFG.events.stage_dt_scene1 * k) for t0, n in EVENTS_1]
    scene2 = [make_event(t0 * k, n, CFG.events.stage_dt_scene2 * k) for t0, n in EVENTS_2]
    events_all = scene1 + scene2
    tscene2 = TL.scene2_start * k
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    from matplotlib.ticker import FuncFormatter

    def style(ax):
        if lang == "ru":
            fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "\u2212"))
            ax.xaxis.set_major_formatter(fmt)
        for s in ax.spines.values():
            s.set_color(tuple(ST.spine_colour))
            s.set_linewidth(ST.spine_width * sc)
        ax.tick_params(colors=tuple(ST.tick_colour), labelsize=ST.tick_size * sc, length=ST.tick_length * sc)

    tt = np.linspace(0, total, TL.pulse_samples)
    for k_ in ids:
        t_film = k_ / fps
        scene = 1 if t_film < tscene2 else 2
        evs = scene1 if scene == 1 else scene2
        fig.texts.clear()
        fig.patches.clear()

        # ------------------------------------------------ tube
        ax_t.clear()
        ax_t.set_facecolor("none")
        ax_t.set_xlim(*TA.xlim)
        ax_t.set_ylim(*TA.ylim)
        ax_t.axis("off")
        bd = TA.body
        ax_t.add_patch(mp.FancyBboxPatch((bd["x"], bd["y"]), bd["w"], bd["h"], boxstyle=f"round,pad=0,rounding_size={bd['rounding']}",
                                         fc=tuple(bd["face"]), ec=tuple(bd["edge"]), lw=bd["lw"] * sc, zorder=1))
        ax_t.plot([CATHODE_X] * 2, [Y_MID - TA.cathode_half, Y_MID + TA.cathode_half], color=tuple(ST.cathode), lw=TA.cathode_width * sc, zorder=3)
        ax_t.text(CATHODE_X + TA.cathode_label["dx"], Y_MID + TA.cathode_label["dy"], tx["cathode"], color=tuple(ST.cathode), fontsize=TA.cathode_label["size"] * sc, ha="center")
        for j in range(1, N_DYN + 1):
            p0, p1 = plate_endpoints(j)
            ax_t.plot([p0[0], p1[0]], [p0[1], p1[1]], color=tuple(ST.dynode), lw=TA.dynode_width * sc, solid_capstyle="round", zorder=3)
            ax_t.text(dynode_center(j)[0], dynode_center(j)[1] + (TA.dynode_label["dy_odd"] if j % 2 == 1 else TA.dynode_label["dy_even"]),
                      rf"$D_{{{j}}}$", color=tuple(ST.dynode_label), fontsize=TA.dynode_label["size"] * sc, ha="center", zorder=4)
        ax_t.plot([ANODE_X] * 2, [Y_MID - TA.anode_half, Y_MID + TA.anode_half], color=tuple(ST.anode), lw=TA.anode_width * sc, ls="--", zorder=3)
        ax_t.text(ANODE_X, Y_MID + TA.anode_label["dy"], tx["anode"], color=tuple(ST.anode), fontsize=TA.anode_label["size"] * sc, ha="center")
        ax_t.text(TA.volts_label["x"], TA.volts_label["y"], tx["volts"], color=DIM, fontsize=TA.volts_label["size"] * sc)

        active = None
        for ev in evs:
            if t_film >= ev.t0:
                active = ev
            # photons entering the window
            tp = t_film - ev.t0
            PHN = TA.photon
            if 0.0 <= tp < TL.photon_duration * k:
                s = tp / (TL.photon_duration * k)
                for i in range(min(ev.n_pe, PHN["max"])):
                    y0 = Y_MID + (i - (min(ev.n_pe, PHN["max"]) - 1) / 2) * PHN["spacing"]
                    xs_ = np.linspace(PHN["x0"], PHN["x0"] + PHN["length"] * s, PHN["points"])
                    ax_t.plot(xs_, y0 + PHN["amplitude"] * np.sin(2 * math.pi * (xs_ * PHN["frequency"])), color=tuple(ST.photon), lw=PHN["width"] * sc, zorder=6)
            # dynode flashes
            for j in range(1, N_DYN + 1):
                if ev.counts[j - 1] <= 0:
                    continue
                FL_ = TA.flash
                arr = ev.stage_time(j - 1) + FL_["arrive_fraction"] * ev.stage_dt
                if t_film >= arr:
                    glow = math.exp(-(t_film - arr) / (FL_["time_factor"] * ev.stage_dt))
                    p0, p1 = plate_endpoints(j)
                    ax_t.plot([p0[0], p1[0]], [p0[1], p1[1]], color=(*ST.flash, min(1.0, glow)), lw=FL_["gain_width"] * sc * glow + FL_["base_width"], zorder=2, solid_capstyle="round")
            # electrons in flight
            for j, fl in enumerate(ev.flights):
                if len(fl.depart) == 0:
                    continue
                inflight = (t_film >= fl.depart) & (t_film < fl.arrive)
                if not inflight.any():
                    continue
                s = (t_film - fl.depart[inflight]) / (fl.arrive[inflight] - fl.depart[inflight])
                pts = bezier(fl.a[inflight], fl.b[inflight], fl.c[inflight], s)
                trail = bezier(fl.a[inflight], fl.b[inflight], fl.c[inflight], np.clip(s - TA.trail["back"], 0, 1))
                for p_, q_ in zip(pts, trail):
                    ax_t.plot([q_[0], p_[0]], [q_[1], p_[1]], color=(*ST.electron_trail, TA.trail["alpha"]), lw=TA.trail["width"] * sc, zorder=5)
                ax_t.scatter(pts[:, 0], pts[:, 1], s=(TA.electron_size * sc) ** 2, c=[tuple(ST.electron)], linewidths=0, zorder=7)

        # counter
        if active is not None:
            reached = 0
            for j in range(N_DYN + 1):
                if t_film >= active.stage_time(j) + TA.flash["arrive_fraction"] * active.stage_dt:
                    reached = j + 1
            reached = min(reached, N_DYN)
            now = active.counts[reached] if reached <= N_DYN else active.counts[-1]
            label = tx["pe"] if reached == 0 else tx["at_dyn"].format(j=reached)
            ax_t.text(TA.counter["x"], TA.counter["y"], rf"{label}: ${now:,}$".replace(",", r"\,"),
                      color=TXT, fontsize=TA.counter["size"] * sc, va="top", zorder=9)
        ax_t.text(TA.constants_label["x"], TA.constants_label["y"], rf"$\delta={num(DELTA, 'g', lang)}\quad N={N_DYN}\quad \delta^N\approx 10^{{{int(round(math.log10(DELTA ** N_DYN)))}}}$",
                  color=DIM, fontsize=TA.constants_label["size"] * sc, va="top", ha="right")

        # ------------------------------------------------ bars
        ax_b.clear()
        ax_b.set_facecolor("none")
        style(ax_b)
        ax_b.set_yscale("log")
        ax_b.set_xlim(-BP.xlim_margin, N_DYN + BP.xlim_margin)
        n_pe_max = max(e.n_pe for e in evs)
        ax_b.set_ylim(BP.ymin, max(BP.ymax_floor, n_pe_max * DELTA ** N_DYN) * BP.ymax_factor)
        ax_b.set_xlabel(tx["stage"], color=DIM, fontsize=BP.label_size * sc)
        ax_b.set_title(tx["per_stage"], color=TXT, fontsize=BP.title_size * sc, loc="left")
        if active is not None:
            mean = mean_counts(active.n_pe)
            ax_b.plot(range(N_DYN + 1), mean, color=(*ST.mean_line, BP.mean_alpha), lw=BP.mean_width * sc, ls="--")
            ax_b.text(N_DYN + BP.mean_label["dx"], mean[-1] * BP.mean_label["factor"], r"$n_0\,\delta^{\,j}$", color=(*ST.mean_line, BP.mean_label["alpha"]), fontsize=BP.mean_label["size"] * sc, ha="right")
            for j in range(N_DYN + 1):
                if t_film >= active.stage_time(j) - BP.bar_time_margin and active.counts[j] > 0:
                    ax_b.bar(j, active.counts[j], color=(*ST.bar, BP.bar_alpha), width=BP.bar_width)
        # ------------------------------------------------ anode pulses
        ax_p.clear()
        ax_p.set_facecolor("none")
        style(ax_p)
        t_lo, t_hi = (0.0, tscene2) if scene == 1 else (tscene2, total)
        ymax = max(e.counts[-1] for e in evs) * PP.ymax_factor
        sig = TL.pulse_width_factor * max(e.stage_dt for e in evs)
        trace = np.zeros_like(tt)
        for ev in evs:
            trace += pulse_shape(tt, ev.anode_time + TL.pulse_centre * ev.stage_dt, ev.counts[-1], sig * TL.pulse_width_gain)
        mask = (tt <= t_film) & (tt >= t_lo) & (tt <= t_hi)
        ax_p.fill_between(tt[mask], 0, trace[mask], color=(*ST.anode, PP.fill_alpha), lw=0)
        ax_p.plot(tt[mask], trace[mask], color=tuple(ST.anode), lw=PP.width * sc)
        ax_p.set_xlim(t_lo, t_hi)
        ax_p.set_ylim(0, ymax)
        ax_p.set_xlabel(r"$t$", color=DIM, fontsize=PP.label_size * sc)
        ax_p.set_title(tx["signal"], color=TXT, fontsize=PP.title_size * sc, loc="left")
        ax_p.set_xticks([])
        if scene == 2:
            for ev in evs:
                if t_film >= ev.anode_time + TL.label_after * ev.stage_dt:
                    ax_p.text(ev.anode_time + TL.pulse_centre * ev.stage_dt, ev.counts[-1] * PP.pe_label["dy"], rf"${ev.n_pe}$ {tx['pe_label']}", color=TXT,
                              fontsize=PP.pe_label["size"] * sc, ha="center", va="bottom")
        elif active is not None and t_film >= active.anode_time + TL.label_after * active.stage_dt:
            ax_p.text(active.anode_time + TL.pulse_centre * active.stage_dt + PP.gain_label["dx"] * active.stage_dt, active.counts[-1] * PP.gain_label["factor"],
                      rf"{tx['gain']} $={sci(active.counts[-1] / active.n_pe, lang)}$",
                      color=TXT, fontsize=PP.gain_label["size"] * sc, ha="right", va="top")
        ax_p.tick_params(labelleft=False)

        # ------------------------------------------------ captions
        cap = tx["cap1"] if scene == 1 else tx["cap2"]
        ttl = tx["title1"] if scene == 1 else tx["title2"]
        fig.text(*LY.caption_pos, cap, color=TXT, fontsize=LY.caption_size * sc)
        fig.text(*LY.title_pos, ttl, color=TXT, fontsize=LY.title_size * sc)
        fade_io = min(smooth(t_film, 0.0, V.fade_s), 1.0 - smooth(t_film, total - V.fade_s, total))
        if abs(t_film - tscene2) < TL.scene_fade_s * k:
            fade_io = min(fade_io, TL.scene_fade_floor + (1.0 - TL.scene_fade_floor) * abs(t_film - tscene2) / (TL.scene_fade_s * k))
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
    ap.add_argument("--seconds", type=float, default=CFG.timeline.film_length)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"pmt_multiplication_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
