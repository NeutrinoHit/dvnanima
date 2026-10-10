#!/usr/bin/env python3
"""Render the 4K films only inside a night window, in resumable chunks.

    python3 render_night.py                       # waits for 22:00, works until 08:00, then stops; run it again the next evening to continue
    python3 render_night.py --start 23:00 --end 07:00 --workers 2
    python3 render_night.py --status              # which chunks exist (the folder media/render_<W>x<H>/)

The film is cut into chunks of frames (media/4k/<lang>_NNN.mp4); a chunk is written under a temporary name and renamed when it is complete, so stopping at the end of the
window or a reboot loses at most the chunks that were running.  When every chunk of a language exists, they are joined into media/wormhole_camera_<lang>_4k.mp4.
The machine has to stay awake (the script holds it with `caffeinate` while it runs): keep it on the power adapter and do not close the lid.
Every number of the film itself is read from config.toml through wormhole_camera.py (--set options are forwarded)."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def hm(text: str) -> dt.time:
    h, m = text.split(":")
    return dt.time(int(h), int(m))


def window_end(now: dt.datetime, start: dt.time, end: dt.time) -> dt.datetime | None:
    """The end of the window that contains `now`, or None if `now` is outside every window."""
    today_start = dt.datetime.combine(now.date(), start)
    today_end = dt.datetime.combine(now.date(), end)
    if start > end:                               # a window over midnight
        if now >= today_start:
            return today_end + dt.timedelta(days=1)
        if now < today_end:
            return today_end
        return None
    return today_end if today_start <= now < today_end else None


def next_start(now: dt.datetime, start: dt.time) -> dt.datetime:
    t = dt.datetime.combine(now.date(), start)
    return t if t > now else t + dt.timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="22:00")
    ap.add_argument("--end", default="08:00")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunks", type=int, default=60, help="chunks per language")
    ap.add_argument("--langs", nargs="+", default=["ru", "en"])
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="forwarded to wormhole_camera.py (for example sky.width=12288)")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    os.environ.setdefault("DVN_NIGHT", "1")
    sets = [f"video.width={args.width}", f"video.height={args.height}", *args.set]
    fwd = [x for s in sets for x in ("--set", s)]
    # the number of frames and the total time come from the configuration
    import tomllib
    cfg = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    total = cfg["timeline"]["part_bounds"][-1] + cfg["timeline"]["title_card"]
    frames = int(round(total * cfg["video"]["fps"]))
    cuts = [round(i * frames / args.chunks) for i in range(args.chunks + 1)]
    tag = "4k" if args.height == 2160 else f"{args.height}p"
    out_dir = HERE / "media" / f"render_{args.width}x{args.height}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def chunk_path(lang: str, i: int) -> Path:
        return out_dir / f"{lang}_{i:03d}.mp4"

    jobs = [(lang, i) for lang in args.langs for i in range(args.chunks) if not chunk_path(lang, i).exists()]
    if args.status:
        for lang in args.langs:
            done = sum(chunk_path(lang, i).exists() for i in range(args.chunks))
            print(f"{lang}: {done}/{args.chunks} chunks")
        return

    start, end = hm(args.start), hm(args.end)
    # keep the machine awake while this script lives
    subprocess.Popen(["caffeinate", "-i", "-m", "-w", str(os.getpid())])
    running: list[tuple[subprocess.Popen, str, int, Path, float]] = []
    est_chunk_s = 0.0
    while jobs or running:
        now = dt.datetime.now()
        deadline = window_end(now, start, end)
        if deadline is None:
            if running:                                   # the window is over: stop the running chunks
                for entry in running:
                    entry[0].terminate()
                for p, lang, i, tmp, t0 in running:
                    p.wait()
                    tmp.unlink(missing_ok=True)
                    jobs.append((lang, i))
                running = []
            wake = next_start(now, start)
            print(f"{now:%H:%M} outside the window, waiting for {wake:%d.%m %H:%M} ({len(jobs)} chunks left)", flush=True)
            time.sleep(min(600, max(5, (wake - now).total_seconds())))
            continue
        # finished chunks
        for entry in list(running):
            p, lang, i, tmp, t0 = entry
            if p.poll() is not None:
                running.remove(entry)
                if p.returncode == 0:
                    est_chunk_s = max(est_chunk_s, time.time() - t0)
                    tmp.rename(chunk_path(lang, i))
                    print(f"{dt.datetime.now():%H:%M} chunk {lang} {i + 1}/{args.chunks} done", flush=True)
                else:
                    tmp.unlink(missing_ok=True)
                    jobs.append((lang, i))
                    print(f"{dt.datetime.now():%H:%M} chunk {lang} {i} failed, will be retried", flush=True)
                    time.sleep(30)
        # new chunks (not when there is no time left for one)
        while jobs and len(running) < args.workers and (est_chunk_s == 0.0 or (deadline - dt.datetime.now()).total_seconds() > est_chunk_s * 1.2):
            lang, i = jobs.pop(0)
            tmp = out_dir / f"{lang}_{i:03d}.part.mp4"
            cmd = ["nice", "-n", str(args.nice), sys.executable, str(HERE / "wormhole_camera.py"), "--lang", lang, "--out", str(tmp), "--range", str(cuts[i]), str(cuts[i + 1]),
                   "--seconds", str(total), *fwd]
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running.append((p, lang, i, tmp, time.time()))
        if not running and jobs:                          # no time left in this window
            print(f"{dt.datetime.now():%H:%M} the window is closing, {len(jobs)} chunks left", flush=True)
            time.sleep(60)
            continue
        time.sleep(20)
    # join the finished languages
    for lang in args.langs:
        parts = [chunk_path(lang, i) for i in range(args.chunks)]
        if all(p.exists() for p in parts):
            lst = out_dir / f"{lang}.txt"
            lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
            target = HERE / "media" / f"wormhole_camera_{lang}_{tag}.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(target)], check=True)
            lst.unlink()
            print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
