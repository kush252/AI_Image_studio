---
title: AI Image Studio
emoji: 🖼️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Image Processing Platform

A classical image-processing platform in two halves:

- **Studio** — a manual editor with **239 operators** across 13 families
  (inspect, tone, colour, denoise, sharpen, frequency, geometric, morphology,
  edges, features, segmentation, augmentation, compose). 80 of them are
  syllabus topics; the rest are included where they do the job better.
- **Advisor** — measures the image across nine quality dimensions, reports
  what falls short *with the number that triggered it*, derives corrective
  parameters by documented formula, and then **re-measures to verify** that the
  correction actually helped.

**No learned models are used anywhere.** Every decision comes from statistics
computed on the single input image at request time. See
[Method notes](#method-notes-no-learned-models).

---

## Quick start

```bash
# 1. Backend
python -m venv .venv
.venv/Scripts/python -m pip install -r backend/requirements.txt   # Windows
#  .venv/bin/python -m pip install -r backend/requirements.txt    # macOS/Linux

# 2. Frontend
cd frontend && npm install && cd ..

# 3. Build the web bundle (served by the API at http://127.0.0.1:8000)
cd frontend && npm run build && cd ..

# 4. Run
cd backend && ../.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

Open **http://127.0.0.1:8000**. There are seven built-in sample images, so you
can explore without uploading anything.

For frontend development with hot reload, run the API as above and then
`cd frontend && npm run dev` (http://localhost:5173, which proxies `/api`).

With Docker: `docker compose up --build`, then http://localhost:5173.

`make help` lists the shortcuts.

---

## How it is built

Four decisions shape everything else. They are recorded in full in
[PLAN.md](PLAN.md); the short version:

**1. Operators are a registry, not endpoints.** Each operator is a declarative
spec — id, category, parameter list with types and bounds, plus its summary,
LaTeX formula and references. `GET /api/operations` serialises the catalogue and
the React client renders *every* control from that schema. Adding an operator
means adding one Python file and no frontend code at all. A test fails the build
if any operator lacks documentation, which is what keeps Learn mode honest as
the catalogue grows.

**2. The pipeline is non-destructive, with prefix caching.** The server always
recomputes from the decoded original, but caches every intermediate under
`sha1(image_id | target | ops[0..i])`. Moving the last slider recomputes exactly
one operator; the rest is a cache hit. The cache is a byte-bounded LRU that
spills to disk rather than dropping work.

**3. Previews run on a proxy; export runs at full resolution.** Interactive work
happens on a ≤1600 px proxy. Parameters that are pixel lengths are declared
`scale_aware` and rescaled on export, so the exported file matches the preview
you approved. Normalised-frequency cutoffs scale *inversely*. A dedicated test
renders recipes at two resolutions and asserts they agree — this is the easiest
thing in a project like this to get wrong.

**4. Everything is `float32` until encode.** Decode once to `float32` in [0,1],
quantise only when writing a file. Operators that model a physical process
(blur, resize, exposure, vignette) declare `domain="linear"` and the executor
converts around them.

### Layout

```
backend/app/
  core/      registry, params, pipeline, cache, store, imageio, masks, blend
  ops/       the 239 operators, one module per family
  analysis/  the measurement suite and the scoring table
  advisor/   rules, parameter derivation, recipe ordering, verification
  reporting/ the lab-report generator
frontend/src/
  components/  ui, canvas, controls (schema-driven), charts, pipeline, advisor
  features/    studio, advisor, augment
docs/        operations-catalogue.md, advisor-spec.md, api.md
```

---

## The Advisor

The pipeline is **measure → score → diagnose → derive → order → apply →
verify**, specified in [docs/advisor-spec.md](docs/advisor-spec.md).

What makes it auditable rather than a list of plausible suggestions:

- **Every finding cites its evidence.** Not "looks a bit dark" but *"mean
  luminance is 0.21, below the 0.35–0.60 range; only 0.1% of pixels are near
  white, so there is headroom to raise exposure without clipping."*
- **Every parameter carries its derivation.** The UI shows, for instance,
  `stops = log2(0.45 / 0.21) = 1.10, clamped to ±2.5` — and why exposure was
  chosen over gamma.
- **The ordering is justified.** Denoise runs before any contrast boost because
  contrast amplifies noise; sharpening runs last because it amplifies whatever
  precedes it.
- **Noise *type* decides the filter**, not just its magnitude: impulse noise
  goes to an adaptive median, periodic interference to an auto-placed notch
  filter, chroma-dominant noise to a chroma-only denoise that leaves luma
  untouched.
- **It closes the loop.** After applying, it re-measures and reports the
  per-dimension delta, separating genuine regressions from inherent trade-offs
  (denoising *necessarily* costs high-frequency detail — that is the operation,
  not a fault). If a dimension regressed, it can name the operator most likely
  responsible by re-running with each one disabled.
- **It knows when to say nothing.** On a clean image it reports no findings at
  all. A diagnostic tool that always finds something wrong is not diagnosing.

### Measured results

Against a 20-case evaluation set built from synthetic degradations of clean
plates, so the ground truth is known exactly
([backend/tests/degradations.py](backend/tests/degradations.py)):

| | |
|---|---|
| Planted defects detected | **20 / 20** |
| Cases where the recipe improved overall health | **17 / 20** |
| Dimension regressions beyond tolerance | **4**, all ≤ 5 points |
| Findings on an unmodified plate | **0** |

The three cases that do not improve are informative rather than broken. Lifting
a dark frame's exposure amplifies the noise that was hiding in its shadows, so
the noise score honestly falls — the noise was always there and is now visible.
The verification panel surfaces exactly that, instead of claiming a win.

---

## Notable implementation details

A few places where the obvious approach was wrong and the code says why:

- **SURF is hand-written.** It is on the syllabus but lives in OpenCV's nonfree
  module, absent from PyPI wheels. Rather than silently substitute a different
  detector, `feat.surf_approx` implements the paper's approximate-Hessian scheme
  directly — integral image, box-filter second derivatives, scale-space
  non-maximum suppression. AKAZE sits beside it for comparison.
- **Noise estimation uses a kernel with a known analytic gain** (Immerkaer's),
  restricted to the flattest tiles — and the tiles are selected from a
  *smoothed* copy, because ranking them on the noisy image preferentially picks
  tiles whose noise happens to be quiet and biases the estimate low.
- **Over-sharpening is detected as a signed edge-profile overshoot**, the way
  ISO 12233 defines it. Absolute overshoot cannot tell a halo from noise, and
  normalising by flat-area variation turns the metric into a blur detector —
  both were tried first.
- **Periodic interference is found with a morphological top-hat on the
  spectrum**, scored by how *isolated* a peak is against its own neighbourhood.
  Radial statistics fail here: a strong blur empties the high-frequency rings,
  collapsing their variance so that numerical dust scores as interference.
- **Uneven illumination is a model fit**, not a variance ratio. `std/mean` is
  inflated by a dark frame; a Michelson ratio survives exposure but not a
  contrast stretch; the share of contrast that is large-scale does not
  discriminate at all, because a 1/f image is mostly low-frequency by
  definition. What distinguishes real uneven lighting is its *shape*.
- **Motion blur length comes from a cepstral period estimate** averaged over a
  strip of spectrum lines. Reading the sinc nulls as local minima of a single
  line over-reported the length roughly fourfold.
- **LUTs are interpolated, not nearest-indexed**, or an identity curve would
  not be an identity operation and 16-bit sources would lose precision.
- **The distance transform uses `DIST_MASK_PRECISE`.** OpenCV's default masks
  are chamfer approximations, a few percent off, and the operator advertises
  Euclidean distance.

---

## Testing

```bash
make test     # 472 tests
make bench    # per-operator timings
```

The suite covers more than smoke tests:

- **Every operator** runs with its defaults and must return finite values in
  range; identity parameters must be exact no-ops.
- **Transform round-trips** (DFT, DCT, Hadamard, Walsh, DWT, Laplacian pyramid)
  must be exact to 1e-5.
- **Scale invariance**: recipes rendered at proxy and at 2.5× must agree. Linear
  and frequency-domain operators are held to SSIM > 0.97; nonlinear rank and
  morphological operators to 0.88, because order statistics genuinely do not
  commute with resampling.
- **A heuristic sweep** flags any new parameter that looks like a pixel length
  but is not marked `scale_aware`, with documented exemptions.
- **Metrics against ground truth**: a flat field reports zero noise; a known σ
  is estimated within 18%; a planted motion blur's angle is recovered within 8°;
  a planted sinusoid is localised in frequency; a tilted grid's angle within
  0.4°.
- **Denoisers must improve PSNR** against the clean reference at σ = 5, 15, 30.
- **Advisor contract**: every fix targets a real operator, every derived
  parameter validates, every finding quotes a number, the recipe is canonically
  ordered, and no clean plate produces a serious finding.

Timings at 1024 px: median **56 ms**, and every operator labelled `fast` is
within the 150 ms interactive budget. The cost labels are derived from
measurement, not guessed.

---

## Method notes: no learned models

Nothing here loads pretrained weights or performs inference. Three families of
algorithm are worth naming explicitly, because they are sometimes mistaken for
machine learning:

- **k-means, mean-shift and SLIC** are unsupervised clustering algorithms,
  fitted to the one image being processed, with no training set and no
  parameters stored between requests.
- **SIFT, SURF, Harris, HOG, LBP and Gabor** are hand-designed operators whose
  coefficients come from their published definitions, not from data.
- **Richardson–Lucy and Wiener deconvolution** are classical inverse filters;
  the PSF is estimated from the image's own spectrum.

The dependency list contains no machine-learning framework: NumPy, OpenCV,
SciPy, scikit-image, PyWavelets and Pillow only.

The **report generator** (`POST /api/report`, or *Generate lab report* in the
UI) writes this declaration into every report alongside the recipe actually
applied, each operator's theory and parameters, the measured before/after
metrics and the Advisor's findings — so the write-up is produced from a real
run rather than transcribed.

---

## Deployment

One container serves both the API and the UI: Vite builds into
`backend/app/static`, and FastAPI mounts that directory at `/` with SPA
fallback. There is nothing separate to host for the frontend.

```bash
docker build -t ipcv-platform .      # from the project ROOT, not backend/
docker run -p 8000:8000 ipcv-platform
```

Then open <http://localhost:8000> — the UI and `/api/*` are the same origin,
so no CORS configuration is needed.

The root `Dockerfile` compiles the frontend itself in a Node stage. Do not use
`backend/Dockerfile` alone for this: it copies whatever happens to be sitting
in `app/static`, so the image silently ships a stale UI, or none at all.

### Where it can run

Anything that takes a Dockerfile: Render, Railway, Fly.io, Google Cloud Run,
Azure Container Apps. `$PORT` is honoured, so the hosts that inject one work
unmodified.

**Not serverless.** Vercel, Lambda and similar are ruled out three times over:
the dependencies are 433 MB against a 250 MB function limit (SciPy 132 MB,
OpenCV 127 MB, NumPy 53 MB, scikit-image 29 MB); a full Advisor diagnose
measures ~15 s on a 1600 px image; and the image store and prefix cache live in
process memory, so a follow-up request landing on another instance cannot see
the uploaded image.

### Sizing and limits

| Setting | Default | Note |
|---|---|---|
| `CACHE_MAX_BYTES` | 1.5 GB in compose | Lower it on a small instance |
| `PREVIEW_MAX_EDGE` | 1600 | Preview resolution; raise costs latency quadratically |
| `MAX_MEGAPIXELS` | 40 | Upload ceiling |
| `CORS_ORIGINS` | localhost only | Only needed if the UI is served from another domain |

Give it ~2 GB of RAM. Health check: `GET /api/health`.

### Scaling

Run **one worker per container**. The image store and render cache are
module-level singletons holding NumPy arrays in process memory, so a second
worker is a second store — an upload handled by one process is invisible to the
other. Measured with `--workers 2`, fetching one image id returned
`200 404 404 404 200 200 404 200`.

Little throughput is lost: OpenCV and NumPy already parallelise across cores
inside one process. To scale past one core, run more **containers** behind a
load balancer with session affinity, or move the store to Redis first. On Cloud
Run, also set minimum instances to 1 and enable session affinity, or the store
is wiped between requests.

---

## Documentation

| | |
|---|---|
| [PLAN.md](PLAN.md) | Architecture decisions, phased build plan, acceptance criteria |
| [docs/operations-catalogue.md](docs/operations-catalogue.md) | All operators with parameters and notes |
| [docs/advisor-spec.md](docs/advisor-spec.md) | Metric formulas, thresholds, rule table, derivations |
| [docs/api.md](docs/api.md) | HTTP API reference |

Keyboard shortcuts: `space` holds the original, `[` / `]` step through the
stack, `ctrl+Z` / `ctrl+shift+Z` undo and redo, `v` / `c` / `b` switch compare
modes, `0` fits to the window, `l` toggles Learn mode.
