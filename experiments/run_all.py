"""Run every experiment in order and summarise the verdicts.

Order matters: the de-risking experiments are ordered by how fast each one can
kill the project, and 5.1 is the kill criterion.  Each script asserts its own
claims, so a failure here is a real disagreement with the mathematics, not a
reporting bug.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from _common import RESULTS, ROOT  # noqa: I001

SCRIPTS = [
    ("5.1  anisotropy (kill criterion)", "exp_5_1_anisotropy.py"),
    ("5.2  exactness horizon", "exp_5_2_horizon.py"),
    ("5.3  smoother / backward pass", "exp_5_3_smoother.py"),
    ("5.4  certification (10^4 orbits)", "exp_5_4_certification.py"),
    ("5.5  slack vs minimal lattice", "exp_5_5_slack.py"),
    ("5.6  baseline comparison", "exp_5_6_baseline.py"),
    ("§3   headline plots", "exp_headline.py"),
    ("§4.6 probabilistic (1D, noisy digits)", "exp_probabilistic.py"),
    ("THEOREM.md §3: the constant C", "exp_theorem.py"),
    ("THEOREM.md §5.3: aperiodic windows", "exp_aperiodic_window.py"),
]


def main() -> int:
    here = Path(__file__).resolve().parent
    failures = []
    for title, script in SCRIPTS:
        print(f"\n{'=' * 72}\n{title}  ({script})\n{'=' * 72}", flush=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(here / script)], cwd=ROOT)
        dt = time.time() - t0
        print(f"[{'ok' if proc.returncode == 0 else 'FAILED'}  {dt:.1f}s]", flush=True)
        if proc.returncode != 0:
            failures.append(script)

    print(f"\n{'=' * 72}\nVERDICTS\n{'=' * 72}")
    for path in sorted(RESULTS.glob("*.json")):
        payload = json.loads(path.read_text())
        verdict = payload.get("verdict")
        if verdict:
            print(f"\n{path.stem}:\n  {verdict}")

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nAll experiments passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
