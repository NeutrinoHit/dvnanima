r"""Thomson's cathode-ray tube (1897): the charge-to-mass ratio from crossed electric and magnetic fields.

The electrons are accelerated in a gun, fly through a region of length ell with an electric field E
(between two plates) and a magnetic field B (perpendicular to the page), and hit a fluorescent screen at the
distance D behind it.  The trajectories are integrated exactly (Boris scheme), nothing is drawn by hand.
Units: e/m = 1, lengths in arbitrary units.  The electron has a negative charge; E points down and B into
the page, so the electric force is up and the magnetic force is down.

  * Only E:   the spot moves up,   y_E =  (e/m) E ell (ell/2 + D) / v^2.
  * Only B:   the spot moves down, y_B = -(e/m) B ell (ell/2 + D) / v      (small angle).
  * Both:     the spot comes back to the centre when eE = evB, that is v = E/B.
  * Then, from the deflection y_E measured with B switched off,
        e/m = y_E v^2 / (E ell (ell/2 + D)) = y_E E / (B^2 ell (ell/2 + D)).
Thomson found e/m of the order of 10^11 C/kg, about 1800 times that of the hydrogen ion: the cathode rays are
light particles, much lighter than an atom.

Film time (s), 52 s:  0-7 no fields; 7-18 E; 18-29 B; 29-39 both, B is swept through the balance;
39-52 balance, then B off and the measurement of e/m.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python thomson_tube.py --lang en            # film -> media/thomson_tube_en.mp4
    python thomson_tube.py --lang ru
    python thomson_tube.py --lang en --snapshot 14
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
_P, _S = CFG.physics, CFG.schedule

TOTAL = _S.total
V0 = _P.v0                      # speed after the gun
E0 = _P.e0                      # electric field strength of the film (arbitrary units, e/m = 1)
B_BAL = E0 / V0                 # balance: v = E / B
X_CATH, X_AN = _P.x_cathode, _P.x_anode         # cathode and anode
X_P0, X_P1 = _P.x_plates_start, _P.x_plates_end # the region with E and B (plate length ell = 3)
X_SCR = _P.x_screen             # screen
ELL = X_P1 - X_P0
DIST = X_SCR - X_P1
LEVER = ELL / 2 + DIST          # ell/2 + D
GUN_ACC = V0 ** 2 / (2 * (X_AN - X_CATH))
RATE = _P.emission_rate         # emitted electrons per second
DT = 1.0 / _P.steps_per_s


# -------------------------------------------------------------- field schedule

def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def fields(t: float) -> tuple[float, float]:
    """(E, B) as functions of the film time."""
    e = E0 * (smooth(t, *_S.e_on) - smooth(t, *_S.e_off) + smooth(t, *_S.e_on_again))
    if t < _S.sweep_start:
        b = B_BAL * (smooth(t, *_S.b_on) - smooth(t, *_S.b_off))
    elif t < _S.sweep_end:                             # B swept through the balance, then fine-tuned back to it
        b = B_BAL * (_S.sweep_base + _S.sweep_gain * smooth(t, *_S.sweep_up) - _S.sweep_back * smooth(t, *_S.sweep_down)) * smooth(t, *_S.sweep_ramp)
    else:
        b = B_BAL * (1 - smooth(t, *_S.b_off_final))
    return float(e), float(b)


# -------------------------------------------------------------------- physics

def step(state: np.ndarray, e_field: float, b_field: float, dt: float = DT) -> None:
    """Advance electrons (columns x, y, vx, vy) by dt.  Boris scheme; e/m = 1, charge negative, E = -E y_hat,
    B = -B z_hat, so a = +E y_hat + B (vy, -vx)."""
    x, y, vx, vy = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    in_gun = (x >= X_CATH) & (x < X_AN)
    in_plates = (x >= X_P0) & (x < X_P1)
    ax = np.where(in_gun, GUN_ACC, 0.0)
    ey = np.where(in_plates, e_field, 0.0)
    bz = np.where(in_plates, b_field, 0.0)
    vx = vx + 0.5 * dt * ax
    vy = vy + 0.5 * dt * ey
    # rotation by the magnetic field: a = B (vy, -vx) -> angular velocity -B
    th = -bz * dt
    c, s = np.cos(th), np.sin(th)
    vx, vy = c * vx - s * vy, s * vx + c * vy
    vx = vx + 0.5 * dt * ax
    vy = vy + 0.5 * dt * ey
    x = x + vx * dt
    y = y + vy * dt
    state[:, 0], state[:, 1], state[:, 2], state[:, 3] = x, y, vx, vy


def screen_deflection(e_field: float, b_field: float) -> float:
    """y at the screen of an electron launched on the axis, constant fields (exact integration)."""
    state = np.array([[X_CATH, 0.0, 0.0, 0.0]])
    for _ in range(_P.screen_steps):
        step(state, e_field, b_field)
        if state[0, 0] >= X_SCR:
            break
    return float(state[0, 1])


def deflection_electric(e_field: float, v: float = V0) -> float:
    return e_field * ELL * LEVER / v ** 2


def deflection_magnetic(b_field: float, v: float = V0) -> float:
    return -b_field * ELL * LEVER / v


def charge_to_mass(y_e: float, e_field: float, b_field: float) -> float:
    """e/m from the deflection with E only and the field B that balances E."""
    return y_e * e_field / (b_field ** 2 * ELL * LEVER)


class Beam:
    """The electrons in flight and the hits on the screen; advanced in film time."""

    def __init__(self, seed: int = _P.seed) -> None:
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.state = np.zeros((0, 4))
        self.hits: list[tuple[float, float]] = []
        self.credit = 0.0

    def advance(self, t_new: float) -> None:
        while self.t < t_new - 1e-9:
            dt = min(DT, t_new - self.t)
            self.credit += RATE * dt
            n_new = int(self.credit)
            self.credit -= n_new
            if n_new:
                fresh = np.zeros((n_new, 4))
                fresh[:, 0] = X_CATH
                fresh[:, 1] = self.rng.normal(0.0, _P.spread, n_new)
                self.state = np.vstack([self.state, fresh])
            e, b = fields(self.t)
            step(self.state, e, b, dt)
            self.t += dt
            if len(self.state):
                hit = self.state[:, 0] >= X_SCR
                if hit.any():
                    self.hits.extend((self.t, float(yh)) for yh in self.state[hit, 1])
                    self.state = self.state[~hit]
                lost = (np.abs(self.state[:, 1]) > _P.lost_y)
                if lost.any():
                    self.state = self.state[~lost]


TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def num(x: float, fmt: str, lang: str) -> str:
    """A number for use inside $...$ (decimal comma in Russian)."""
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, SC, FV, LY, ST = CFG.video, CFG.scene, CFG.front_view, CFG.layout, CFG.style
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = ST.background
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)
    TXT = tuple(ST.text)
    DIM = tuple(ST.dim)
    col_e = tuple(ST.electron)          # electrons
    col_E = tuple(ST.electric)          # plates, electric field
    col_B = tuple(ST.magnetic)          # magnetic field
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes(SC.axes, facecolor="none")
    fv = fig.add_axes(FV.axes, facecolor="none")    # front view of the screen
    k = total / TOTAL
    beam = Beam()
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    # static decor
    for k_ in ids:
        t_film = k_ / fps
        t = t_film / k
        beam.advance(t)
        e_now, b_now = fields(t)
        fig.texts.clear()
        ax.clear()
        ax.set_xlim(*SC.xlim)
        ax.set_ylim(*SC.ylim)
        ax.set_aspect("auto")
        ax.axis("off")
        # glass envelope
        env = SC.envelope
        ax.add_patch(mp.FancyBboxPatch((env["x"], env["y"]), env["w"], env["h"], boxstyle=f"round,pad=0,rounding_size={env['rounding']}",
                                       fc=tuple(env["face"]), ec=tuple(env["edge"]), lw=env["lw"] * sc, zorder=1))
        # cathode and anode with the slit
        ax.plot([X_CATH, X_CATH], [-SC.cathode["half"], SC.cathode["half"]], color=tuple(ST.cathode), lw=SC.cathode["width"] * sc, zorder=3, solid_capstyle="round")
        ax.add_patch(mp.Circle((X_CATH, 0), SC.cathode["glow"], color=(*ST.cathode, ST.glow_alpha), zorder=2))
        for yy0, yy1 in SC.anode_plates:
            ax.plot([X_AN, X_AN], [yy0, yy1], color=tuple(ST.anode), lw=SC.anode_width * sc, zorder=3)
        for yy0, yy1 in SC.focus_plates:
            ax.plot([SC.focus_x, SC.focus_x], [yy0, yy1], color=(*ST.anode, SC.focus_alpha), lw=SC.focus_width * sc, zorder=3)
        ax.text(X_CATH, SC.cathode_label["dy"], tx["cathode"], color=DIM, fontsize=SC.cathode_label["size"] * sc, ha="center")
        ax.text(X_AN + SC.anode_label["dx"], SC.anode_label["y"], tx["anode"], color=DIM, fontsize=SC.anode_label["size"] * sc, ha="center")
        # plates and the electric field
        ea = min(1.0, abs(e_now) / E0)
        for yy, sign in ((SC.plates_y, "+"), (-SC.plates_y, "\u2212")):
            ax.plot([X_P0, X_P1], [yy, yy], color=(*col_E, SC.plate_alpha), lw=SC.plate_width * sc, zorder=3, solid_capstyle="butt")
            for xx in np.linspace(X_P0 + SC.sign_margin, X_P1 - SC.sign_margin, SC.sign_count):
                ax.text(xx, yy + (SC.sign_dy if yy > 0 else -SC.sign_dy), sign, color=(*col_E, SC.sign_alpha[0] + SC.sign_alpha[1] * ea), fontsize=SC.sign_size * sc,
                        ha="center", va="center", zorder=4)
        if ea > SC.field_threshold:
            FA = SC.field_arrows
            for xx in np.linspace(X_P0 + FA["margin"], X_P1 - FA["margin"], FA["count"]):
                ax.annotate("", xy=(xx, -FA["y"]), xytext=(xx, FA["y"]),
                            arrowprops=dict(arrowstyle="-|>", color=(*col_E, SC.arrow_alpha * ea), lw=FA["lw"] * sc, mutation_scale=FA["mutation"] * sc), zorder=2)
            ax.text(X_P1 + SC.e_label["dx"], 0.0, r"$\mathbf{E}$", color=(*col_E, SC.e_label_alpha[0] + SC.e_label_alpha[1] * ea), fontsize=SC.e_label["size"] * sc, va="center")
        # the magnetic field: crosses into the page, pole pieces
        ba = min(1.0, abs(b_now) / B_BAL / SC.b_divisor)
        PO = SC.pole
        ax.add_patch(mp.Rectangle((X_P0, PO["y_top"]), ELL, PO["h"], fc=tuple(PO["face"]), ec=tuple(PO["edge"]), lw=PO["lw"] * sc, zorder=2))
        ax.add_patch(mp.Rectangle((X_P0, PO["y_bottom"]), ELL, PO["h"], fc=tuple(PO["face"]), ec=tuple(PO["edge"]), lw=PO["lw"] * sc, zorder=2))
        if ba > SC.field_threshold:
            CR = SC.crosses
            ca = CR["alpha_gain"] * ba + CR["alpha_base"]
            for xx in np.linspace(X_P0 + CR["margin"], X_P1 - CR["margin"], CR["count"]):
                for yy in CR["rows"]:
                    h_ = CR["half"]
                    ax.add_patch(mp.Circle((xx, yy), CR["radius"], fill=False, ec=(*col_B, ca), lw=CR["lw"] * sc, zorder=2))
                    ax.plot([xx - h_, xx + h_], [yy - h_, yy + h_], color=(*col_B, ca), lw=CR["lw"] * sc, zorder=2)
                    ax.plot([xx - h_, xx + h_], [yy + h_, yy - h_], color=(*col_B, ca), lw=CR["lw"] * sc, zorder=2)
            ax.text(X_P0 + ELL / 2, SC.b_label["y"], r"$\mathbf{B}\ \otimes$", color=(*col_B, 0.5 + 0.5 * ba), fontsize=SC.b_label["size"] * sc, ha="center")
        ax.text(X_P0 + ELL / 2, SC.apparatus_label["y"], tx["plates"] + "  +  " + tx["coil"], color=DIM, fontsize=SC.apparatus_label["size"] * sc, ha="center")
        # screen and ruler
        ax.plot([X_SCR, X_SCR], [-SC.screen["half"], SC.screen["half"]], color=(*ST.screen, SC.screen["alpha"]), lw=SC.screen["width"] * sc, zorder=2)
        RU = SC.ruler
        for yy in np.arange(RU["lo"], RU["hi"], RU["step"]):
            ax.plot([X_SCR + RU["x0"], X_SCR + RU["x1_major"] if abs(yy) % 1 < 1e-9 else X_SCR + RU["x1_minor"]], [yy, yy], color=(*ST.ruler, RU["alpha"]), lw=1 * sc)
        ax.text(X_SCR + SC.screen_label["dx"], SC.screen_label["y"], tx["screen"], color=DIM, fontsize=SC.screen_label["size"] * sc, ha="center")
        # electrons
        if len(beam.state):
            ax.scatter(beam.state[:, 0], beam.state[:, 1], s=(SC.electron_size * sc * 72 / dpi) ** 2, c=[(*col_e, SC.electron_alpha)], linewidths=0, zorder=6)
            ax.scatter(beam.state[:, 0], beam.state[:, 1], s=(SC.electron_halo_size * sc * 72 / dpi) ** 2, c=[(*col_e, SC.electron_halo_alpha)], linewidths=0, zorder=5)
        # the spot (persistence)
        recent = [(th, yh) for th, yh in beam.hits[-SC.spot_history:] if t - th < SC.spot_recent]
        if recent:
            ys_ = np.array([yh for _, yh in recent])
            al = np.array([math.exp(-(t - th) / SC.spot_decay) for th, _ in recent])
            for scale, a_ in SC.spot_layers:
                ax.scatter(np.full(len(ys_), X_SCR), ys_, s=(scale * sc * 72 / dpi) ** 2,
                           facecolors=[(*ST.spot, min(1.0, a_ * al_)) for al_ in al], linewidths=0, zorder=7)
        # front view: the screen face
        fv.clear()
        fv.set_facecolor("none")
        fv.set_xlim(-FV.lim, FV.lim)
        fv.set_ylim(-FV.lim, FV.lim)
        fv.set_aspect("equal")
        fv.axis("off")
        aa = np.linspace(0, 2 * math.pi, FV.rim_points)
        fv.plot(FV.radius * np.cos(aa), FV.radius * np.sin(aa), color=(*ST.screen, SC.screen["alpha"]), lw=FV.rim_width * sc)
        TK = FV.ticks
        for yy in np.arange(TK["lo"], TK["hi"], TK["step"]):
            fv.plot([-TK["half"], TK["half"]], [yy, yy], color=(*ST.ruler, FV.tick_alpha), lw=TK["lw"] * sc)
        XC = FV.cross
        fv.plot([-XC["half"], XC["half"]], [0, 0], color=(*ST.front_cross, XC["alpha"]), lw=XC["lw"] * sc)
        fv.plot([0, 0], [-XC["half"], XC["half"]], color=(*ST.front_cross, XC["alpha"]), lw=XC["lw"] * sc)
        if recent:
            for scale, a_ in FV.spot_layers:
                fv.scatter(np.zeros(len(ys_)), ys_, s=(scale * sc * 72 / dpi) ** 2,
                           facecolors=[(*ST.spot, min(1.0, a_ * al_)) for al_ in al], linewidths=0, zorder=7)
        hw, hn = SC.spot_average_window, SC.spot_average_hits
        y_spot = float(np.mean([yh for th, yh in beam.hits[-hn:] if t - th < hw])) if any(t - th < hw for th, _ in beam.hits[-hn:]) else None
        fig.text(*FV.title_pos, tx["front"], color=TXT, fontsize=FV.title_size * sc)
        if y_spot is not None:
            fig.text(*FV.value_pos, rf"$y={num(y_spot, '+.2f', lang)}$".replace("+", "+").replace("-", "\u2212"), color=tuple(ST.spot), fontsize=FV.value_size * sc, ha="center")

        # readouts of the fields
        fig.text(*LY.readout_e_pos, rf"$E={num(e_now, '.2f', lang)}$", color=col_E, fontsize=LY.readout_size * sc)
        fig.text(*LY.readout_b_pos, rf"$B={num(b_now, '.2f', lang)}$", color=col_B, fontsize=LY.readout_size * sc)
        fig.text(*LY.units_pos, tx["units"], color=DIM, fontsize=LY.units_size * sc)

        # formulas and captions by stage
        T1, T2, T3, T4, T5, T6 = _S.stages
        if t < T1:
            cap, fml = tx["cap0"], [tx["f0"]]
        elif t < T2:
            cap, fml = tx["capE"], [tx["fE"]]
        elif t < T3:
            cap, fml = tx["capB"], [tx["fE"], tx["fB"]]
        elif t < T4:
            cap, fml = tx["capS"], [tx["fE"], tx["fB"]]
        elif t < T5:
            cap, fml = tx["capBal"], [tx["fBal"]]
        elif t < T6:
            cap, fml = tx["capM"], [tx["fBal"], tx["fM"]]
        else:
            cap, fml = tx["capEnd"], [tx["fM"]]
        for i, f_ in enumerate(fml):
            fig.text(LY.formula_x, LY.formula_y - LY.formula_row * i, f_, color=TXT, fontsize=LY.formula_size * sc)
        if t >= _S.measure_from:
            # measured quantities
            y_e = deflection_electric(E0)
            if t >= _S.result_from:
                em = charge_to_mass(screen_deflection(E0, 0.0), E0, B_BAL)
                fig.text(*LY.result_pos, rf"$e/m={num(em, '.2f', lang)}$   ({tx['meas']})", color=tuple(ST.measured), fontsize=LY.result_size * sc)
                fig.text(*LY.reference_pos, tx["res"], color=DIM, fontsize=LY.reference_size * sc, linespacing=LY.reference_linespacing)
        if _S.balance_window[0] <= t < _S.balance_window[1]:
            # the balance marker
            bal = abs(b_now - B_BAL) < _S.balance_tolerance or (_S.balance_hold[0] <= t < _S.balance_hold[1])
            if bal:
                fig.text(*LY.balance_pos, rf"$v=E/B={num(E0 / B_BAL, '.1f', lang)}$", color=tuple(ST.measured), fontsize=LY.balance_size * sc)
        fig.text(*LY.title_pos, tx["title"], color=tuple(ST.title_color), fontsize=LY.title_size * sc)
        fig.text(*LY.subtitle_pos, tx["sub"], color=DIM, fontsize=LY.subtitle_size * sc)
        fig.patches.append(matplotlib.patches.Rectangle((0, 0), *LY.caption_box, transform=fig.transFigure, color=tuple(LY.caption_box_colour), zorder=0, lw=0))
        fig.text(*LY.caption_pos, cap, color=TXT, fontsize=LY.caption_size * sc)
        fade_io = min(smooth(t_film, 0.0, V.fade_s), 1.0 - smooth(t_film, total - V.fade_s, total))
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)
        if fade_io < 1.0:
            frame = bg_rgba + (frame - bg_rgba) * fade_io
        frame = frame.clip(0, 255).astype(np.uint8)
        fig.patches.clear()
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
    out = args.out or HERE / "media" / f"thomson_tube_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
