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

Usage:
    python pmt_multiplication.py                 # film -> media/pmt_multiplication.mp4
    python pmt_multiplication.py --preview
    python pmt_multiplication.py --snapshot 12
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

HERE = Path(__file__).resolve().parent

DELTA = 4.0                   # mean secondary-emission yield per dynode
N_DYN = 10                    # number of dynodes
MAX_DRAWN = 36                # electrons drawn per stage
CATHODE_X = 1.7
ANODE_X = 14.1
DYN_X0, DYN_DX = 3.5, 1.05
Y_TOP, Y_BOT, Y_MID = 7.7, 2.3, 5.0
PLATE_HALF = 0.5
EVENTS_1 = ((1.0, 1),)                                    # (start time s, photoelectrons) in scene 1
EVENTS_2 = ((21.0, 2), (26.0, 6), (30.5, 12))             # scene 2
SEEDS = {1: 5, 2: 11, 6: 3, 12: 7}


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
    a = math.radians(38.0) * (1 if j % 2 == 1 else -1)
    d = np.array([math.cos(a), math.sin(a)]) * PLATE_HALF
    return np.array([cx, cy]) - d, np.array([cx, cy]) + d


def node_point(j: int, rng: np.random.Generator) -> np.ndarray:
    """Random emission/landing point of stage j: 0 photocathode, 1..N dynode, N+1 anode."""
    if j == 0:
        return np.array([CATHODE_X + 0.15 * rng.random(), Y_MID + 2.4 * (rng.random() - 0.5) * 2 * 0.9])
    if j == N_DYN + 1:
        return np.array([ANODE_X, Y_MID + 2.7 * (rng.random() - 0.5) * 2 * 0.9])
    p0, p1 = plate_endpoints(j)
    s = 0.15 + 0.7 * rng.random()
    inward = np.array([0.0, -0.12 if j % 2 == 1 else 0.12])
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
        return self.t0 + 0.9 + j * self.stage_dt

    @property
    def anode_time(self) -> float:
        return self.stage_time(N_DYN + 1) - 0.12 * self.stage_dt


def make_event(t0: float, n_pe: int, stage_dt: float) -> Event:
    rng = np.random.default_rng(SEEDS.get(n_pe, 1))
    counts = multiply(n_pe, rng)
    flights = []
    for j in range(N_DYN + 1):
        m = min(counts[j], MAX_DRAWN)
        a = np.array([node_point(j, rng) for _ in range(m)]).reshape(m, 2)
        b = np.array([node_point(j + 1, rng) for _ in range(m)]).reshape(m, 2)
        mid = 0.5 * (a + b)
        c = mid + np.column_stack([0.25 * (rng.random(m) - 0.5), (Y_MID - mid[:, 1]) * 0.30 + 0.5 * (rng.random(m) - 0.5)])
        t_dep = t0 + 0.9 + j * stage_dt + stage_dt * 0.10 * rng.random(m)
        t_arr = t_dep + stage_dt * (0.80 + 0.07 * rng.random(m))
        flights.append(Flight(a, b, c, t_dep, t_arr))
    return Event(t0, n_pe, counts, stage_dt, flights)


def pulse_shape(t: np.ndarray, t_c: float, height: float, width: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((t - t_c) / width) ** 2)


def smooth(v: float, a: float, b: float) -> float:
    u = min(max((v - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


# --------------------------------------------------------------------- film

def render(out: Path, size: tuple[int, int], fps: int, total: float, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    W, H = size
    dpi = 100
    sc = H / 720.0
    BG = "#03060c"
    TXT = (0.88, 0.93, 1.0, 0.95)
    DIM = (0.6, 0.68, 0.8, 0.9)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax_t = fig.add_axes([0.03, 0.14, 0.65, 0.72], facecolor="none")
    ax_b = fig.add_axes([0.745, 0.585, 0.24, 0.27], facecolor="none")
    ax_p = fig.add_axes([0.745, 0.20, 0.24, 0.27], facecolor="none")

    k = total / 40.0
    scene1 = [make_event(t0 * k, n, 1.45 * k) for t0, n in EVENTS_1]
    scene2 = [make_event(t0 * k, n, 0.78 * k) for t0, n in EVENTS_2]
    events_all = scene1 + scene2
    tscene2 = 20.0 * k
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def style(ax):
        for s in ax.spines.values():
            s.set_color((0.5, 0.6, 0.75, 0.6))
            s.set_linewidth(0.8 * sc)
        ax.tick_params(colors=(0.65, 0.72, 0.85), labelsize=9 * sc, length=3 * sc)

    tt = np.linspace(0, total, 1600)
    for k_ in ids:
        t_film = k_ / fps
        scene = 1 if t_film < tscene2 else 2
        evs = scene1 if scene == 1 else scene2
        fig.texts.clear()
        fig.patches.clear()

        # ------------------------------------------------ tube
        ax_t.clear()
        ax_t.set_facecolor("none")
        ax_t.set_xlim(-0.8, 16.4)
        ax_t.set_ylim(0.0, 10.1)
        ax_t.axis("off")
        ax_t.add_patch(mp.FancyBboxPatch((0.6, 0.55), 15.0, 8.8, boxstyle="round,pad=0,rounding_size=0.9",
                                         fc=(0.10, 0.18, 0.30, 0.35), ec=(0.6, 0.72, 0.9, 0.6), lw=1.4 * sc, zorder=1))
        ax_t.plot([CATHODE_X] * 2, [Y_MID - 2.6, Y_MID + 2.6], color=(0.45, 0.75, 1.0), lw=4.0 * sc, zorder=3)
        ax_t.text(CATHODE_X + 0.1, Y_MID + 2.9, "photocathode", color=(0.45, 0.75, 1.0), fontsize=10 * sc, ha="center")
        for j in range(1, N_DYN + 1):
            p0, p1 = plate_endpoints(j)
            ax_t.plot([p0[0], p1[0]], [p0[1], p1[1]], color=(1.0, 0.62, 0.3), lw=4.5 * sc, solid_capstyle="round", zorder=3)
            ax_t.text(dynode_center(j)[0], dynode_center(j)[1] + (0.55 if j % 2 == 1 else -0.75),
                      f"D{j}", color=(1.0, 0.75, 0.5), fontsize=8.5 * sc, ha="center", zorder=4)
        ax_t.plot([ANODE_X] * 2, [Y_MID - 2.8, Y_MID + 2.8], color=(0.55, 1.0, 0.7), lw=2.0 * sc, ls="--", zorder=3)
        ax_t.text(ANODE_X, Y_MID + 3.0, "anode", color=(0.55, 1.0, 0.7), fontsize=10 * sc, ha="center")
        ax_t.text(0.8, 0.1, "≈ 100 V between neighbouring dynodes", color=DIM, fontsize=9.5 * sc)

        active = None
        for ev in evs:
            if t_film >= ev.t0:
                active = ev
            # photons entering the window
            tp = t_film - ev.t0
            if 0.0 <= tp < 0.9 * k:
                s = tp / (0.9 * k)
                for i in range(min(ev.n_pe, 4)):
                    y0 = Y_MID + (i - (min(ev.n_pe, 4) - 1) / 2) * 0.9
                    xs_ = np.linspace(-0.7, -0.7 + 2.3 * s, 60)
                    ax_t.plot(xs_, y0 + 0.13 * np.sin(2 * math.pi * (xs_ * 3.0)), color=(1.0, 0.95, 0.5), lw=1.8 * sc, zorder=6)
            # dynode flashes
            for j in range(1, N_DYN + 1):
                if ev.counts[j - 1] <= 0:
                    continue
                arr = ev.stage_time(j - 1) + 0.9 * ev.stage_dt
                if t_film >= arr:
                    glow = math.exp(-(t_film - arr) / (0.45 * ev.stage_dt))
                    p0, p1 = plate_endpoints(j)
                    ax_t.plot([p0[0], p1[0]], [p0[1], p1[1]], color=(1.0, 0.9, 0.6, min(1.0, glow)), lw=9 * sc * glow + 1, zorder=2, solid_capstyle="round")
            # electrons in flight
            for j, fl in enumerate(ev.flights):
                if len(fl.depart) == 0:
                    continue
                inflight = (t_film >= fl.depart) & (t_film < fl.arrive)
                if not inflight.any():
                    continue
                s = (t_film - fl.depart[inflight]) / (fl.arrive[inflight] - fl.depart[inflight])
                pts = bezier(fl.a[inflight], fl.b[inflight], fl.c[inflight], s)
                trail = bezier(fl.a[inflight], fl.b[inflight], fl.c[inflight], np.clip(s - 0.12, 0, 1))
                for p_, q_ in zip(pts, trail):
                    ax_t.plot([q_[0], p_[0]], [q_[1], p_[1]], color=(0.5, 0.85, 1.0, 0.5), lw=1.2 * sc, zorder=5)
                ax_t.scatter(pts[:, 0], pts[:, 1], s=(4.2 * sc) ** 2, c=[(0.75, 0.95, 1.0, 1.0)], linewidths=0, zorder=7)

        # counter
        if active is not None:
            reached = 0
            for j in range(N_DYN + 1):
                if t_film >= active.stage_time(j) + 0.9 * active.stage_dt - 0.0 * active.stage_dt:
                    reached = j + 1
            reached = min(reached, N_DYN)
            now = active.counts[reached] if reached <= N_DYN else active.counts[-1]
            label = "photoelectrons" if reached == 0 else f"electrons at dynode {reached}"
            ax_t.text(-0.6, 10.0, f"{label}:  {now:,}".replace(",", " "),
                      color=TXT, fontsize=12 * sc, va="top", zorder=9)
        ax_t.text(16.2, 10.0, rf"$\delta={DELTA:g}$   $N={N_DYN}$   $\delta^N\approx 10^{{{int(round(math.log10(DELTA ** N_DYN)))}}}$",
                  color=DIM, fontsize=10.5 * sc, va="top", ha="right")

        # ------------------------------------------------ bars
        ax_b.clear()
        ax_b.set_facecolor("none")
        style(ax_b)
        ax_b.set_yscale("log")
        ax_b.set_xlim(-0.6, N_DYN + 0.6)
        n_pe_max = max(e.n_pe for e in evs)
        ax_b.set_ylim(0.6, max(2e6, n_pe_max * DELTA ** N_DYN) * 4)
        ax_b.set_xlabel("stage j (0 = photocathode)", color=DIM, fontsize=9.5 * sc)
        ax_b.set_title("electrons per stage", color=TXT, fontsize=10.5 * sc, loc="left")
        if active is not None:
            mean = mean_counts(active.n_pe)
            ax_b.plot(range(N_DYN + 1), mean, color=(1, 1, 1, 0.55), lw=1.0 * sc, ls="--")
            ax_b.text(N_DYN + 0.4, mean[-1] * 0.28, r"$n_0\,\delta^{\,j}$", color=(1, 1, 1, 0.7), fontsize=9.5 * sc, ha="right")
            for j in range(N_DYN + 1):
                if t_film >= active.stage_time(j) - 0.02 and active.counts[j] > 0:
                    ax_b.bar(j, active.counts[j], color=(1.0, 0.72, 0.3, 0.9), width=0.7)
        # ------------------------------------------------ anode pulses
        ax_p.clear()
        ax_p.set_facecolor("none")
        style(ax_p)
        t_lo, t_hi = (0.0, tscene2) if scene == 1 else (tscene2, total)
        ymax = max(e.counts[-1] for e in evs) * 1.15
        sig = 0.10 * max(e.stage_dt for e in evs)
        trace = np.zeros_like(tt)
        for ev in evs:
            trace += pulse_shape(tt, ev.anode_time + 0.6 * ev.stage_dt, ev.counts[-1], sig * 1.5)
        mask = (tt <= t_film) & (tt >= t_lo) & (tt <= t_hi)
        ax_p.fill_between(tt[mask], 0, trace[mask], color=(0.55, 1.0, 0.7, 0.28), lw=0)
        ax_p.plot(tt[mask], trace[mask], color=(0.55, 1.0, 0.7), lw=1.6 * sc)
        ax_p.set_xlim(t_lo, t_hi)
        ax_p.set_ylim(0, ymax)
        ax_p.set_xlabel("t", color=DIM, fontsize=9.5 * sc)
        ax_p.set_title("anode signal", color=TXT, fontsize=10.5 * sc, loc="left")
        ax_p.set_xticks([])
        if scene == 2:
            for ev in evs:
                if t_film >= ev.anode_time + 0.9 * ev.stage_dt:
                    ax_p.text(ev.anode_time + 0.6 * ev.stage_dt, ev.counts[-1] * 1.04, f"{ev.n_pe} PE", color=TXT,
                              fontsize=9.5 * sc, ha="center", va="bottom")
        elif active is not None and t_film >= active.anode_time + 0.9 * active.stage_dt:
            ax_p.text(active.anode_time + 0.6 * active.stage_dt, active.counts[-1] * 1.04,
                      f"gain = {active.counts[-1] / active.n_pe:.2g}".replace("e+0", "e").replace("e+", "e"),
                      color=TXT, fontsize=9.5 * sc, ha="center", va="bottom")
        ax_p.tick_params(labelleft=False)

        # ------------------------------------------------ captions
        if scene == 1:
            cap = ("one photoelectron: each electron knocks out several secondary electrons from the next dynode, the number grows as δ^j",
                   "один фотоэлектрон: каждый электрон выбивает из следующего динода несколько вторичных, число растёт как δ^j")
            ttl = ("Photomultiplier tube", "Фотоэлектронный умножитель")
        else:
            cap = ("a scintillation flash of more light gives more photoelectrons: the pulse height measures the number of photons (the energy)",
                   "вспышка сцинтиллятора с большим светом даёт больше фотоэлектронов: амплитуда импульса измеряет число фотонов (энергию)")
            ttl = ("Photomultiplier tube: pulse height", "ФЭУ: амплитуда импульса")
        fig.text(0.03, 0.058, cap[0].replace("δ^j", "$\\delta^{\\,j}$"), color=TXT, fontsize=10.8 * sc)
        fig.text(0.03, 0.022, cap[1].replace("δ^j", "$\\delta^{\\,j}$"), color=DIM, fontsize=9.6 * sc)
        fig.text(0.03, 0.915, ttl[0], color=TXT, fontsize=16 * sc)
        fig.text(0.03, 0.885, ttl[1], color=DIM, fontsize=11 * sc)
        fade_io = min(smooth(t_film, 0.0, 0.5), 1.0 - smooth(t_film, total - 0.5, total))
        if abs(t_film - tscene2) < 0.5 * k:
            fade_io = min(fade_io, 0.2 + 0.8 * abs(t_film - tscene2) / (0.5 * k))
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)
        if fade_io < 1.0:
            bg = np.array([3, 6, 12, 255], np.float32)
            frame = bg + (frame - bg) * fade_io
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
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--out", type=Path, default=HERE / "media" / "pmt_multiplication.mp4")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(args.out.with_suffix(".png"), (1280, 720), 30, args.seconds, snap=args.snapshot)
    elif args.preview:
        render(args.out.with_name("pmt_multiplication_preview.mp4"), (640, 360), 15, args.seconds)
    else:
        render(args.out, (1280, 720), 30, args.seconds)


if __name__ == "__main__":
    main()
