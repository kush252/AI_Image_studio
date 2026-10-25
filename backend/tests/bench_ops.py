"""Per-operator benchmark. PLAN.md Phase 10.

Not part of the test suite -- run it directly:

    python -m tests.bench_ops            # proxy resolution
    python -m tests.bench_ops --full      # also time at 2048 px
    python -m tests.bench_ops --category smooth
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import registry  # noqa: E402
from app.core.pipeline import OpInvocation, execute  # noqa: E402

SKIP = {
    "tone.hist_match", "color.transfer", "geom.register_ecc",
    "geom.register_phase", "feat.match", "feat.pyramid_blend", "feat.template",
    "feat.stitch", "aug.cutmix", "aug.mixup", "compose.blend",
}


def make_image(edge: int) -> np.ndarray:
    from tests.fixtures.synth import plate

    return plate(0, h=edge, w=edge)


def time_op(img: np.ndarray, op_id: str, repeats: int = 3) -> float | None:
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        try:
            execute(img, [OpInvocation(op_id, {})], image_id="", use_cache=False,
                    ctx={"original": img})
        except Exception:
            return None
        dt = (time.perf_counter() - t0) * 1000.0
        best = dt if best is None else min(best, dt)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also time at 2048 px")
    ap.add_argument("--category", default=None)
    ap.add_argument("--top", type=int, default=0, help="show only the N slowest")
    args = ap.parse_args()

    registry.load_all()
    ops = [s for s in registry.all_ops() if s.id not in SKIP]
    if args.category:
        ops = [s for s in ops if s.category == args.category]

    sizes = [1024] + ([2048] if args.full else [])
    images = {e: make_image(e) for e in sizes}

    print(f"{len(ops)} operators, {' and '.join(f'{e}px' for e in sizes)}, "
          f"best of 3\n")
    header = f"{'operator':<34}{'cost':<8}" + "".join(f"{e}px".rjust(11) for e in sizes)
    print(header)
    print("-" * len(header))

    rows = []
    for spec in ops:
        times = [time_op(images[e], spec.id) for e in sizes]
        rows.append((spec, times))

    if args.top:
        rows.sort(key=lambda r: -(r[1][0] or 0))
        rows = rows[: args.top]

    for spec, times in rows:
        cells = "".join(
            (f"{t:9.1f}  " if t is not None else "     skip  ") for t in times
        )
        print(f"{spec.id:<34}{spec.cost:<8}{cells}")

    ok = [(s, t[0]) for s, t in rows if t[0] is not None]
    if ok:
        print()
        print(f"median {np.median([t for _, t in ok]):.1f} ms, "
              f"slowest {max(ok, key=lambda x: x[1])[0].id} "
              f"at {max(t for _, t in ok):.0f} ms")
        # The interactive budget from PLAN.md: a slider move should stay under
        # 150 ms at proxy resolution.
        over = [(s.id, t) for s, t in ok if t > 150 and s.cost == "fast"]
        if over:
            print(f"\n{len(over)} operator(s) declared 'fast' exceed the 150 ms "
                  f"interactive budget at 1024 px:")
            for name, t in sorted(over, key=lambda x: -x[1]):
                print(f"   {name}: {t:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
