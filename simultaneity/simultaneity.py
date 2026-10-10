r"""The relativity of simultaneity on a Minkowski diagram.

Three flashes A, B, C happen at t = 0 at x = -1, 0, +1 in the frame O (c = 1).  In a frame O' that moves with
the velocity v along x they happen at

    t'_i = gamma (t_i - v x_i) = - gamma v x_i,       x'_i = gamma (x_i - v t_i) = gamma x_i,

so for v > 0 the order is C, B, A and for v < 0 it is A, B, C.  On the diagram the axes of O' are the lines
x = v ct (the ct' axis) and ct = v x (the x' axis); the lines of simultaneity of O' are parallel to the x' axis:
ct = v x + ct'/gamma.  A dashed line from an event along its line of simultaneity to the ct' axis shows how
t'_i is read off.  The two small panels show what happens in each frame: the flashes with their expanding
wave fronts (a view from above), each in the time of its own frame.

The last part: two events separated by a *spacelike* interval (|dx| > |dt|) change their order for some
velocities (the line of simultaneity of O' sweeps through the pair), two events separated by a *timelike*
interval (|dx| < |dt|) never do, because the line of simultaneity cannot leave the region outside the light
cone.  Therefore causality is not violated.

Film time (s), 56 s, one pass = the clocks of both frames run from -0.9 to +0.9:
0-5 introduction; 5.5-12.5 v = 0; ramp; 15-24 v = +0.3; ramp; 26.5-35.5 v = -0.5; 37.5-52 timelike and
spacelike pairs while v sweeps from -0.6 to +0.6; 52-56 summary.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python simultaneity.py --lang en            # film -> media/simultaneity_en.mp4
    python simultaneity.py --lang ru
    python simultaneity.py --lang en --snapshot 20
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
EVENTS = {k: tuple(v) for k, v in CFG.events.flashes.to_dict().items()}          # (x, ct) in O
PAIRS = {k: tuple(v) for k, v in CFG.events.pairs.to_dict().items()}             # the last part
T_SWEEP = CFG.events.clock_sweep                                                  # clocks run over [-T_SWEEP, +T_SWEEP]
LATE_START = CFG.timeline.late_start


def gamma(v: float) -> float:
    return 1.0 / math.sqrt(1.0 - v * v)


def boost(x: float, t: float, v: float) -> tuple[float, float]:
    """(x', t') of the event (x, t) in the frame that moves with v (c = 1)."""
    g = gamma(v)
    return g * (x - v * t), g * (t - v * x)


def order(events: dict[str, tuple[float, float]], v: float) -> str:
    """Names of the events sorted by their time t' in the frame with velocity v."""
    return "".join(sorted(events, key=lambda n: boost(*events[n], v)[1]))


def interval2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (b[1] - a[1]) ** 2 - (b[0] - a[0]) ** 2


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


# velocity and clock schedule -------------------------------------------------------------

PASSES = tuple(tuple(p) for p in CFG.timeline.passes)


def velocity(t: float) -> float:
    V_ = CFG.velocity
    if t < V_.still_until:
        return 0.0
    if t < V_.ramp1[1]:
        return V_.v1 * smooth(t, *V_.ramp1)
    if t < V_.hold1_until:
        return V_.v1
    if t < V_.ramp2[1]:
        return V_.v1 + (V_.v2 - V_.v1) * smooth(t, *V_.ramp2)
    if t < V_.hold2_until:
        return V_.v2
    if t < V_.ramp3[1]:
        return V_.v2 + (V_.v3 - V_.v2) * smooth(t, *V_.ramp3)
    if t < CFG.timeline.sweep_end:
        return V_.v3 + V_.sweep_amplitude * smooth(t, *V_.sweep)
    return V_.v_final


def clock(t: float) -> float | None:
    """The common reading of the two clocks (-0.9 ... +0.9) during a pass; None between the passes."""
    for a, b in PASSES:
        if a <= t <= b:
            return -T_SWEEP + 2 * T_SWEEP * (t - a) / (b - a)
    return None


TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def num(x: float, fmt: str, lang: str) -> str:
    s = format(x, fmt).replace("-", "\u2212")
    return s.replace(".", "{,}") if lang == "ru" else s


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, EV, TL, DG, PN, RO, LY, ST = CFG.video, CFG.events, CFG.timeline, CFG.diagram, CFG.panels, CFG.readouts, CFG.layout, CFG.style
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = ST.background
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)
    TXT = tuple(ST.text)
    DIM = tuple(ST.dim)
    COL = {k: tuple(v) for k, v in ST.events.to_dict().items()}
    COL_OP = tuple(ST.frame_o)           # axes and lines of the frame O'
    AX_C = tuple(ST.axis_colour)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    dg = fig.add_axes(DG.axes, facecolor="none")
    s1 = fig.add_axes(PN.axes_o, facecolor="none")
    s2 = fig.add_axes(PN.axes_op, facecolor="none")
    k = total / TOTAL
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def rgba(c, a):
        return (*c[:3], max(0.0, min(1.0, a)))

    for k_ in ids:
        t_film = k_ / fps
        t = t_film / k
        v = velocity(t)
        g = gamma(v)
        late = t >= LATE_START
        events = PAIRS if late else EVENTS
        cl = clock(t)
        fig.texts.clear()
        fig.patches.clear()

        # ------------------------------------------------ diagram
        dg.clear()
        dg.set_facecolor("none")
        dg.set_xlim(*DG.xlim)
        dg.set_ylim(*DG.ylim)
        dg.set_aspect("equal")
        dg.axis("off")
        # O axes
        XA, TA_ = DG.x_axis, DG.t_axis
        dg.annotate("", xy=(XA["x"], 0), xytext=(-XA["x"], 0), arrowprops=dict(arrowstyle="-|>", color=(*AX_C, XA["alpha"]), lw=XA["lw"] * sc, mutation_scale=XA["mutation"] * sc))
        dg.annotate("", xy=(0, TA_["y_top"]), xytext=(0, TA_["y_bottom"]), arrowprops=dict(arrowstyle="-|>", color=(*AX_C, XA["alpha"]), lw=XA["lw"] * sc, mutation_scale=XA["mutation"] * sc))
        dg.text(DG.x_label["x"], DG.x_label["dy"], tx["x"], color=TXT, fontsize=DG.x_label["size"] * sc, ha="right")
        dg.text(DG.t_label["dx"], DG.t_label["y"], tx["ct"], color=TXT, fontsize=DG.t_label["size"] * sc, va="top")
        # the light cone of the origin
        CN = DG.cone
        for sgn in (1, -1):
            dg.plot([0, sgn * CN["length_up"]], [0, CN["length_up"]], color=(1, 1, 1, CN["alpha"]), lw=CN["width"] * sc, ls=(0, tuple(CN["dash"])))
            dg.plot([0, sgn * CN["length_down"]], [0, -CN["length_down"]], color=(1, 1, 1, CN["alpha"]), lw=CN["width"] * sc, ls=(0, tuple(CN["dash"])))
        # axes of O' (appear with |v|)
        FA = DG.frame_axes
        va = smooth(abs(v), *FA["appear"])
        if va > FA["threshold"]:
            ymax = FA["ymax"]
            dg.plot([-v * (-FA["ymin"]), v * ymax], [FA["ymin"], ymax], color=rgba(COL_OP, FA["alpha"] * va), lw=FA["width"] * sc)          # ct' axis: x = v ct
            xlim = FA["xlim"]
            dg.plot([-xlim, xlim], [-v * xlim, v * xlim], color=rgba(COL_OP, FA["alpha"] * va), lw=FA["width"] * sc)           # x' axis: ct = v x
            FL = DG.frame_labels
            dg.text(v * FL["ct_x"] + (FL["ct_dx"] if v >= 0 else -FL["ct_dx"]), FL["ct_y"], tx["ctp"], color=rgba(COL_OP, va), fontsize=FL["size"] * sc, ha="left" if v >= 0 else "right", va="top")
            dg.text(FL["x_x"], v * FL["x_x"] + (FL["x_dy_pos"] if v >= 0 else FL["x_dy_neg"]), tx["xp"], color=rgba(COL_OP, va), fontsize=FL["size"] * sc, ha="right")
        # lines of simultaneity
        if cl is not None:
            # O: horizontal
            dg.plot([-DG.simultaneity_o["x"], DG.simultaneity_o["x"]], [cl, cl], color=(*AX_C, DG.simultaneity_o["alpha"]), lw=DG.simultaneity_o["width"] * sc)
            # O': ct = v x + ct'/gamma
            xx = np.array([-DG.simultaneity_o["x"], DG.simultaneity_o["x"]])
            dg.plot(xx, v * xx + cl / g, color=rgba(COL_OP, DG.simultaneity_op["alpha"]), lw=DG.simultaneity_op["width"] * sc)
        # events
        names = list(events)
        for name in names:
            ex, et = events[name]
            xp_, tp_ = boost(ex, et, v)
            col = COL[name]
            # t' read-off: dashed line along the line of simultaneity to the ct' axis
            RD = DG.readoff
            if va > FA["threshold"] and not late:
                px, pt = g * v * tp_, g * tp_
                dg.plot([ex, px], [et, pt], color=rgba(COL_OP, RD["alpha"] * va), lw=RD["width"] * sc, ls=(0, tuple(RD["dash"])))
                dg.plot([px], [pt], "o", color=rgba(COL_OP, va), ms=RD["marker"] * sc)
                dg.text(px + (RD["label_dx"] if v >= 0 else -RD["label_dx"]), pt, rf"$t'_{name}$", color=rgba(COL_OP, va), fontsize=RD["size"] * sc,
                        ha="left" if v >= 0 else "right", va="center")
            # halos when a line of simultaneity has passed the event: white for O, amber for O'
            if cl is not None and not late:
                HA = DG.halo
                for tflash, hc in ((et, (1.0, 1.0, 1.0)), (tp_, COL_OP)):
                    if cl >= tflash:
                        age = cl - tflash
                        al = HA["alpha"] * math.exp(-age / HA["decay"])
                        r0 = HA["r0_op"] + HA["r_op"] * age if hc is COL_OP else HA["r0_o"] + HA["r_o"] * age
                        dg.add_patch(mp.Circle((ex, et), r0, fill=False, ec=rgba(hc, al), lw=HA["width"] * sc, zorder=5))
            ap0, ap1 = EV.appear_range
            appear = smooth(t, ap0 + EV.appear_step * names.index(name), ap1 + EV.appear_step * names.index(name)) if t < TL.intro_end else 1.0
            if late:
                lp0, lp1 = EV.late_appear_range
                appear = smooth(t, lp0 + EV.late_appear_step * names.index(name), lp1 + EV.late_appear_step * names.index(name))
            dg.plot([ex], [et], "o", color=rgba(col, appear), ms=DG.event_marker["size"] * sc * (DG.event_marker["min_scale"] + (1 - DG.event_marker["min_scale"]) * appear), zorder=6)
            dg.text(ex, et + DG.event_label["dy"], rf"${name}$", color=rgba(col, appear), fontsize=DG.event_label["size"] * sc, ha="center", va="top", zorder=7)
        if late:
            # the light cone of P is the cone of the origin; label it
            LL = DG.light_label
            dg.text(LL["x"], LL["y"], tx["light"], color=(1, 1, 1, LL["alpha"]), fontsize=LL["size"] * sc, ha="right")
            # the line of simultaneity of O' through P (t' = 0)
            xx = np.array([-DG.late_op["x"], DG.late_op["x"]])
            dg.plot(xx, v * xx, color=rgba(COL_OP, DG.late_op["alpha"]), lw=DG.late_op["width"] * sc)
            for name in ("Q", "R"):                      # the lines of simultaneity of O' through Q and R
                ex, et = events[name]
                dg.plot(xx, v * (xx - ex) + et, color=rgba(COL[name], DG.late_through["alpha"]), lw=DG.late_through["width"] * sc, ls=(0, tuple(DG.late_through["dash"])))

        # ------------------------------------------------ the two panels
        def panel(ax, frame_label, tcur, pos, tflash):
            ax.clear()
            ax.set_facecolor("none")
            ax.set_xlim(-PN.xlim, PN.xlim)
            ax.set_ylim(-PN.ylim, PN.ylim)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.plot([-PN.axis_line["x"], PN.axis_line["x"]], [0, 0], color=(*AX_C, PN.axis_line["alpha"]), lw=PN.axis_line["width"] * sc)
            ax.text(PN.frame_label["x"], PN.frame_label["y"], frame_label, color=TXT, fontsize=PN.frame_label["size"] * sc, va="top")
            if tcur is not None:
                ax.text(PN.time_label["x"], PN.time_label["y"], rf"$t={num(tcur, '+.2f', lang)}$", color=TXT, fontsize=PN.time_label["size"] * sc, ha="right", va="top")
            for name in names:
                x0 = pos[name]
                col = COL[name]
                ax.plot([x0], [0], "o", color=rgba(col, 1.0), ms=PN.marker * sc, zorder=6)
                ax.text(x0, PN.name_label["dy"], rf"${name}$", color=rgba(col, 1.0), fontsize=PN.name_label["size"] * sc, ha="center", va="top")
                if tcur is None:
                    continue
                dtm = tcur - tflash[name]
                if dtm >= 0:
                    r = dtm
                    if r < PN.front_max:
                        al = PN.front_alpha * math.exp(-r / PN.front_decay)
                        ax.add_patch(mp.Circle((x0, 0), max(r, PN.min_radius), fill=False, ec=rgba(col, al), lw=PN.front_width * sc, zorder=4))
                    if dtm < PN.flash_duration:
                        FS = PN.flash
                        ax.add_patch(mp.Circle((x0, 0), FS["r0"] + FS["gain"] * dtm, fc=rgba(tuple(FS["colour"]), FS["alpha"] * (1 - dtm / PN.flash_duration)), ec="none", zorder=7))

        if late:
            for ax_ in (s1, s2):
                ax_.clear()
                ax_.axis("off")
            # in the last part the panels show the table instead
        else:
            pos_O = {n: events[n][0] for n in names}
            tfl_O = {n: events[n][1] for n in names}
            pos_Op = {n: boost(*events[n], v)[0] for n in names}
            tfl_Op = {n: boost(*events[n], v)[1] for n in names}
            panel(s1, tx["O"], cl, pos_O, tfl_O)
            panel(s2, tx["Op"], cl, pos_Op, tfl_Op)

        # ------------------------------------------------ readouts
        fig.text(*RO.formula_pos, tx["formula"] if not late else tx["formula2"], color=TXT, fontsize=RO.formula_size * sc)
        if not late:
            vals = [boost(*events[n], v)[1] for n in names]
            line = RO.times_separator.join(rf"$t'_{n}={num(tp, '+.2f', lang)}$" for n, tp in zip(names, vals))
            fig.text(*RO.times_pos, line, color=TXT, fontsize=RO.times_size * sc)
            od = order(events, v)
            if abs(v) < EV.order_tolerance:
                txt = tx["simul"]
            else:
                txt = r"$" + r"\to ".join(od) + r"$"
            fig.text(*RO.order_pos, f"{tx['order']}:  {txt}", color=COL_OP, fontsize=RO.order_size * sc)
            fig.text(*RO.velocity_pos, rf"$v={num(v, '+.2f', lang)}$", color=COL_OP, fontsize=RO.velocity_size * sc)
        else:
            fig.text(*RO.velocity_pos, rf"$v={num(v, '+.2f', lang)}$", color=COL_OP, fontsize=RO.velocity_size * sc)
            for i, (a, b, kind) in enumerate((("P", "R", "space"), ("P", "Q", "time"))):
                dt_ = boost(*events[b], v)[1] - boost(*events[a], v)[1]
                first = a if dt_ > 0 else b
                lab = tx["dep"] if kind == "space" else tx["always"]
                ds2 = interval2(events[a], events[b])
                y0 = RO.pair_y0 - RO.pair_dy * i
                fig.text(RO.times_pos[0], y0, rf"${a},\,{b}$: {tx[kind]}", color=TXT, fontsize=RO.pair_title_size * sc)
                fig.text(RO.times_pos[0], y0 - RO.interval_dy, rf"$\Delta s^2=\Delta t^2-\Delta x^2={num(ds2, '+.2f', lang)}$", color=DIM, fontsize=RO.interval_size * sc)
                verdict = tx["simul"] if abs(dt_) < EV.simultaneous_tolerance else rf"${first}$ " + tx["first"]
                fig.text(RO.times_pos[0], y0 - RO.verdict_dy, rf"$\Delta t'_{{{a}{b}}}={num(dt_, '+.2f', lang)}\,\to$  {verdict}",
                         color=rgba(COL_OP, RO.verdict_alpha), fontsize=RO.verdict_size * sc)
                fig.text(RO.times_pos[0], y0 - RO.kind_dy, f"{tx['order']}: {lab}", color=DIM, fontsize=RO.kind_size * sc)
        # titles and captions
        fig.text(*LY.title_pos, tx["title"], color=tuple(ST.title_color), fontsize=LY.title_size * sc)
        fig.text(*LY.subtitle_pos, tx["sub"], color=DIM, fontsize=LY.subtitle_size * sc)
        c0, c1, c2, c3, c4, c5, c6 = TL.caption_steps
        if t < c0:
            cap = tx["cap_intro"]
        elif t < c1:
            cap = tx["cap0"]
        elif t < c3:
            cap = tx["cap1"] if t >= c2 else tx["capr"]
        elif t < c4:
            cap = tx["capr"]
        elif t < c5:
            cap = tx["cap2"]
        elif t < c6:
            cap = tx["cap3"]
        else:
            cap = tx["cap4"]
        fig.patches.append(matplotlib.patches.Rectangle((0, 0), *LY.caption_box, transform=fig.transFigure, color=tuple(LY.caption_box_colour), zorder=0, lw=0))
        fig.text(*LY.caption_pos, cap, color=TXT, fontsize=LY.caption_size * sc)
        fade_io = min(smooth(t_film, 0.0, V.fade_s), 1.0 - smooth(t_film, total - V.fade_s, total))
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
    out = args.out or HERE / "media" / f"simultaneity_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
