"""Lab-report generator. PLAN.md Phase 10.

Builds the write-up from a real session: the recipe actually applied, each
operator's theory and parameters, the measured before/after metrics, and the
Advisor's findings. Including the "no learned models" statement with its
justification, because that is the claim an examiner will probe.
"""
from __future__ import annotations

import base64
import datetime as _dt

import numpy as np

from ..analysis import metrics as M
from ..analysis.score import OVERALL_WEIGHTS, SCORING, score_all
from ..core import imageio as iio
from ..core import registry
from ..core.pipeline import OpInvocation, execute

NO_ML_STATEMENT = """\
### Declaration: no learned models are used

Nothing in this project loads pretrained weights or performs inference. Every
decision the Advisor makes is derived from statistics computed from the pixels
of the single input image at request time: histograms and percentiles, moments,
gradient and Laplacian energy, Fourier and wavelet spectra, morphological
residuals, and model fits (radial polynomials for vignetting, low-order
surfaces for illumination, line PSFs for motion blur).

Three families of algorithm are worth distinguishing explicitly, because they
are sometimes mistaken for machine learning:

* **k-means, mean-shift and SLIC** are unsupervised clustering algorithms. They
  are fitted to the one image being processed, have no training set, and store
  no parameters between requests.
* **SIFT, SURF, Harris, HOG, LBP and Gabor** are hand-designed operators whose
  coefficients come from their published definitions, not from data.
* **Richardson-Lucy and Wiener deconvolution** are classical inverse filters;
  the PSF is estimated from the image's own spectrum.

The dependency list contains no machine-learning framework: NumPy, OpenCV,
SciPy, scikit-image, PyWavelets and Pillow only.
"""


def _fmt(v, nd=5):
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".")
    if isinstance(v, (list, tuple)):
        if len(v) > 8:
            return f"[{len(v)} values]"
        return "[" + ", ".join(_fmt(x, 3) for x in v) + "]"
    if isinstance(v, dict):
        return "{...}"
    return str(v)


def _b64(arr: np.ndarray, max_edge: int = 720, quality: int = 85) -> str:
    small, _ = iio.make_proxy(arr, max_edge)
    payload = iio.encode(small, "jpeg", quality)
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


def build_report(
    image,
    ops: list[dict],
    *,
    target: str = "proxy",
    diagnosis: dict | None = None,
    verification: dict | None = None,
    title: str = "Image Processing Report",
    embed_images: bool = True,
) -> dict:
    """Render a markdown report plus the data behind it."""
    base = image.get(target)
    res = execute(base, [OpInvocation.from_dict(o) for o in ops],
                  image_id=image.image_id, target=target,
                  collect_views=False, ctx={"original": base})
    after = res.image

    m_before = M.measure_all(base)
    m_after = M.measure_all(after)
    s_before = score_all(m_before)
    s_after = score_all(m_after)

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L: list[str] = []
    L.append(f"# {title}")
    L.append("")
    L.append(f"*Generated {now} from a live session. Every number below was "
             f"measured, not transcribed.*")
    L.append("")

    # ---------------------------------------------------------------- input
    L.append("## 1. Input")
    L.append("")
    L.append("| Property | Value |")
    L.append("|---|---|")
    L.append(f"| Source | `{image.name or image.image_id}` |")
    L.append(f"| Dimensions | {image.width} x {image.height} "
             f"({image.width * image.height / 1e6:.2f} MP) |")
    L.append(f"| Channels | {image.channels} |")
    L.append(f"| Format | {image.source_format}, "
             f"{image.source_bit_depth}-bit |")
    L.append(f"| Processed at | {after.shape[1]} x {after.shape[0]} "
             f"({target}) |")
    if image.exif:
        interesting = {k: v for k, v in image.exif.items()
                       if k in ("Make", "Model", "ExposureTime", "FNumber",
                                "ISOSpeedRatings", "FocalLength", "LensModel")}
        for k, v in interesting.items():
            L.append(f"| {k} | {v} |")
    L.append("")

    if embed_images:
        L.append("### Before and after")
        L.append("")
        L.append(f"![Original]({_b64(base)})")
        L.append("")
        L.append(f"![Processed]({_b64(after)})")
        L.append("")

    # ------------------------------------------------------------- pipeline
    L.append("## 2. Processing pipeline")
    L.append("")
    if not ops:
        L.append("*No operators were applied.*")
    else:
        L.append(f"{len(ops)} operator(s), applied in order. Total render time "
                 f"{res.total_ms:.1f} ms.")
        L.append("")
        for i, o in enumerate(ops, 1):
            oid = o.get("op_id", "?")
            try:
                spec = registry.get(oid)
            except Exception:
                L.append(f"### {i}. `{oid}` (unknown operator)")
                continue
            L.append(f"### {i}. {spec.name}")
            L.append("")
            L.append(f"`{spec.id}` &mdash; *{spec.category}*"
                     + ("  **[syllabus]**" if spec.syllabus else ""))
            L.append("")
            if spec.summary:
                L.append(spec.summary)
                L.append("")
            if spec.formula_latex:
                L.append("$$")
                L.append(spec.formula_latex)
                L.append("$$")
                L.append("")
            if spec.when_to_use:
                L.append(f"**When to use:** {spec.when_to_use}")
                L.append("")
            params = o.get("params") or {}
            if params:
                L.append("| Parameter | Value |")
                L.append("|---|---|")
                for k, v in params.items():
                    L.append(f"| `{k}` | {_fmt(v)} |")
                L.append("")
            extras = []
            if o.get("blend") and o["blend"] != "normal":
                extras.append(f"blend mode `{o['blend']}`")
            if o.get("opacity", 1.0) < 1.0:
                extras.append(f"opacity {o['opacity']:.2f}")
            if o.get("mask"):
                extras.append(f"masked ({o['mask'].get('kind', 'custom')})")
            if extras:
                L.append("Applied with " + ", ".join(extras) + ".")
                L.append("")
            if spec.references:
                L.append("References: " + "; ".join(spec.references))
                L.append("")

    # -------------------------------------------------------------- metrics
    L.append("## 3. Measured effect")
    L.append("")
    L.append(f"Overall image health: **{s_before['overall']:.1f} -> "
             f"{s_after['overall']:.1f}** "
             f"({s_after['overall'] - s_before['overall']:+.1f}).")
    L.append("")
    L.append("| Dimension | Before | After | Change |")
    L.append("|---|---|---|---|")
    for dim in OVERALL_WEIGHTS:
        b = s_before["dimensions"].get(dim, {}).get("score")
        a = s_after["dimensions"].get(dim, {}).get("score")
        if b is None or a is None:
            continue
        L.append(f"| {dim.capitalize()} | {b:.1f} | {a:.1f} | {a - b:+.1f} |")
    L.append("")

    L.append("### Metric detail")
    L.append("")
    L.append("| Metric | Before | After | Expected range |")
    L.append("|---|---|---|---|")
    for _dim, rows in SCORING.items():
        for group, name, _w, ilo, ihi, _flo, _fhi in rows:
            if name.startswith("_"):
                continue
            b = m_before.get(group, {}).get(name)
            a = m_after.get(group, {}).get(name)
            if not isinstance(b, (int, float)) or not isinstance(a, (int, float)):
                continue
            L.append(f"| `{group}.{name}` | {_fmt(b)} | {_fmt(a)} | "
                     f"{ilo} to {ihi} |")
    L.append("")

    # ------------------------------------------------------------- advisor
    if diagnosis:
        L.append("## 4. Advisor findings")
        L.append("")
        findings = diagnosis.get("findings") or []
        if not findings:
            L.append("No defects crossed their thresholds.")
        for f in findings:
            L.append(f"### {f['title']} &mdash; {f['severity']} "
                     f"(confidence {f['confidence']:.2f})")
            L.append("")
            L.append(f"*Rule `{f['id']}`, dimension `{f['dimension']}`.*")
            L.append("")
            L.append(f["explanation"])
            L.append("")
            ev = f.get("evidence") or {}
            if ev.get("metric") is not None:
                L.append(f"**Evidence:** `{ev['metric']}` = "
                         f"{_fmt(ev.get('value'))}, expected "
                         f"{ev.get('expected')}.")
                L.append("")
            fix = f.get("fix")
            if fix:
                L.append(f"**Proposed fix:** {fix['label']}")
                L.append("")
                for op in fix["ops"]:
                    L.append(f"- `{op['op_id']}` with "
                             + ", ".join(f"`{k}`={_fmt(v)}"
                                         for k, v in (op.get('params') or {}).items()))
                L.append("")
                L.append(f"**Derivation:** {fix['derivation']}")
                L.append("")
            else:
                L.append("*No automatic fix is offered for this finding.*")
                L.append("")
        for n in diagnosis.get("notes") or []:
            L.append(f"> {n}")
            L.append("")

    # -------------------------------------------------------- verification
    if verification:
        L.append("## 5. Verification")
        L.append("")
        L.append(verification.get("summary", ""))
        L.append("")
        if verification.get("regressions"):
            L.append("**Regressions flagged:**")
            L.append("")
            for r in verification["regressions"]:
                line = (f"- `{r['dimension']}` fell {r['delta']:+.1f} "
                        f"({r['before']:.1f} -> {r['after']:.1f})")
                if r.get("likely_cause"):
                    c = r["likely_cause"]
                    line += (f"; most likely `{c['op_id']}`, whose removal "
                             f"recovers {c['recovered']:+.1f}")
                L.append(line)
            L.append("")
        if verification.get("tradeoffs"):
            L.append("**Expected trade-offs** (inherent to the operators used, "
                     "not faults):")
            L.append("")
            for t in verification["tradeoffs"]:
                L.append(f"- `{t['dimension']}` {t['delta']:+.1f} "
                         f"({t['before']:.1f} -> {t['after']:.1f})")
            L.append("")

    # ----------------------------------------------------------- statement
    L.append("## 6. Method notes")
    L.append("")
    L.append(NO_ML_STATEMENT)
    L.append("")
    used = [registry.get(o["op_id"]) for o in ops
            if registry.exists(o.get("op_id", ""))]
    syll = [s for s in used if s.syllabus]
    if used:
        L.append(f"Of the {len(used)} operators applied, {len(syll)} are "
                 f"syllabus topics: "
                 + ", ".join(sorted({s.name for s in syll})) + "."
                 if syll else f"Of the {len(used)} operators applied, none are "
                              f"syllabus topics.")
        L.append("")

    markdown = "\n".join(L)
    return {
        "markdown": markdown,
        "scores_before": s_before,
        "scores_after": s_after,
        "metrics_before": m_before,
        "metrics_after": m_after,
        "operator_count": len(ops),
        "syllabus_operator_count": len(syll),
        "render_ms": res.total_ms,
    }
