"""Shared plumbing for the de-risking experiments.

Every experiment writes one figure and one JSON of summary numbers into
``results/``, and logs every parameter it used into that JSON so the run is
reproducible.  Seeds are fixed at the call site.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Run as plain scripts (`python experiments/exp_*.py`), so put the repo root on
# the path rather than requiring an editable install.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Categorical slots 1-4 of the reference palette, validated as a set for the
# light surface (worst adjacent CVD dE 9.2, normal-vision 27.6).  Aqua sits
# below 3:1 contrast, so every series is *also* direct-labelled -- identity is
# never carried by colour alone.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SERIES = {
    "truth": "#4a3aa7",      # violet
    "naive": "#eb6834",      # orange
    "lattice": "#2a78d6",    # blue
    "filtered": "#1baf7a",   # aqua
}
EXTRA = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#d6d5d0",
        "axes.labelcolor": INK_MUTED,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,   # grid must never draw over the marks
        "grid.color": "#e8e7e2",
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "font.size": 10,
        "axes.titlesize": 12,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "figure.dpi": 140,
    })


def label_end(ax, x, y, text, color, dx=0.4, dy=0.0) -> None:
    """Direct label at the right end of a series (never a number on every point)."""
    ax.annotate(text, xy=(x[-1], y[-1]), xytext=(x[-1] + dx, y[-1] + dy),
                color=color, fontsize=9,
                va="center", ha="left", annotation_clip=False)


def save(fig, name: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _jsonable(o):
    if isinstance(o, Fraction):
        return f"{o.numerator}/{o.denominator}"
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


def write_json(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    payload = dict(payload)
    payload["_meta"] = {
        "experiment": name,
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    path.write_text(json.dumps(payload, indent=2, default=_jsonable) + "\n")
    return path


def report(name: str, lines: list[str]) -> None:
    print(f"\n=== {name} ===")
    for line in lines:
        print(f"  {line}")
