# Image Processing Platform — Build Plan

> This is the implementation contract. `docs/operations-catalogue.md` holds the full
> operator list; `docs/advisor-spec.md` holds the metric formulas, thresholds and
> rule table for Section 2. Read all three before writing code.

---

## 0. What we are building

A web platform with **two** top-level sections, reachable from a left rail:

| Section | Name | Purpose |
|---|---|---|
| 1 | **Studio** | A professional, manual editor. Dozens of classical IPCV operators grouped into sub-sections (Inspect, Tone, Color, Smooth/Denoise, Sharpen/Detail, Frequency, Geometric, Morphology, Edges & Features, Segmentation, Augment). The user drives everything. |
| 2 | **Advisor** | Automatic *image understanding*: measure the image across nine quality dimensions using pure DSP, report where it falls short, and emit one-click fixes with **pre-derived parameters**. |

### Hard constraint: no learned models

Nothing in this project loads pretrained weights or performs inference. Every
decision in the Advisor comes from measurable statistics of the pixel data —
histograms, moments, gradients, spectra, morphological residuals.

k-means, mean-shift and SLIC *are* allowed: they are unsupervised clustering
algorithms fit at request time on the single input image, with no training set
and no stored parameters. SIFT/Harris/HOG/Gabor are hand-designed operators, not
models. This distinction gets stated explicitly in the generated report
(Phase 10) because an examiner will ask.

### Deviations from `gen_idea.md`

| Original | Change | Why |
|---|---|---|
| Streamlit | **FastAPI + React** | Streamlit re-runs the whole script per widget change; 11 operator families with masks, interactive canvases and an op stack is not expressible in it. |
| Single flat feature list | **Operation registry + schema-driven UI** | Adding an operator must mean adding one Python file and zero frontend lines. |
| "Apply each adjustment to the original" | **Non-destructive op stack with prefix caching** | Required both for correctness (no cumulative quantization) and for interactivity. |

---

## 1. Architectural decisions (decide once, now)

These are the decisions everything else depends on. Changing them later is
expensive.

### D1 — Operation registry, not endpoints-per-filter

Every operator is a declarative `OperationSpec`: stable `id`, display name,
category, an ordered list of `ParamSpec`, a pure function
`(img: np.ndarray, **params) -> np.ndarray | OpResult`, plus documentation
fields (`summary`, `formula_latex`, `when_to_use`, `references`).

`GET /api/operations` serialises the whole catalogue. The React client renders
**all** controls from that schema. Consequences:

- One new filter = one Python file + one `@register` decorator.
- Validation lives in one place (`core/params.py`) and is shared by the HTTP
  layer, the Advisor, the augmentation sampler and the test suite.
- Recipes are JSON — portable, diffable, embeddable in the lab report.

### D2 — Non-destructive pipeline with prefix caching

Client state is `(image_id, [ {op_id, params, enabled, opacity, blend, mask} ... ])`.
The server **always recomputes from the decoded original**. To keep this
interactive, cache every intermediate:

```
key_i = sha1(image_id | target_res | canonical_json(ops[0..i]))
```

`PipelineCache` is a byte-bounded LRU of `float32` arrays (default 1.5 GB,
configurable, spilling largest-oldest to `.npy` in the session temp dir). Moving
the last slider hits `key_{n-1}` and recomputes exactly one operator. Reordering
or toggling an early op invalidates only from that index forward.

### D3 — Proxy resolution for preview, full resolution only on export

On upload, build a proxy with the long edge clamped to `PREVIEW_MAX_EDGE`
(default 1600, user-switchable 1024/1600/2048) plus a cached Gaussian pyramid.
All interactive work runs on the proxy. Export re-runs the identical recipe at
native resolution in a background job.

Scale-dependent parameters (kernel sizes, blur sigmas, unsharp radii, morph SE
sizes, DFT cutoffs, CLAHE tile grids) must be declared `scale_aware: true` in
their `ParamSpec`, and the executor rescales them by `full_edge / proxy_edge` on
export. **Getting this wrong means the exported image looks different from the
preview** — the most common bug in this class of app. There is a dedicated test
in Phase 1 that renders a recipe at two resolutions and asserts perceptual
agreement.

### D4 — Work in `float32`, and offer linear light

Decode to `float32` in `[0,1]` once. Every operator consumes and returns
`float32`. Quantise to 8/16-bit only at encode. This kills banding from chained
operators.

Separately, expose a **"work in linear light"** toggle. Operations that model a
physical process — Gaussian/box blur, resize, exposure scaling, add/blend, lens
and vignette correction, defocus simulation — are *wrong* in gamma-encoded sRGB.
Each `OperationSpec` carries `domain: "srgb" | "linear" | "either"`, and the
executor inserts EOTF/OETF conversions around runs of operators that disagree
with the current mode. Default off (matches what users expect from Photoshop and
from textbook examples), on for the Advisor's exposure and blur fixes.

### D5 — Transport: binary renders behind content-addressed URLs

`POST /api/pipeline/preview` returns `{render_id, width, height, stats, timings}`.
The client then loads `GET /api/render/{render_id}` — an immutable
`Cache-Control: immutable` JPEG q88 (or WebP, negotiated). Benefits over base64:
~33% less payload, the browser's own image cache handles A/B and undo for free,
and `<img>` decode happens off the main thread.

Exports and 16-bit output go through a separate `/api/export` job as PNG/TIFF.

### D6 — CPU-bound work off the event loop

OpenCV and NumPy release the GIL. Run every operator through
`anyio.to_thread.run_sync` on a bounded worker pool (`min(4, cpu_count)`
threads), with `cv2.setNumThreads(2)` so one request cannot starve the others.
Heavy jobs (batch augmentation, full-res export, multi-image analysis) go to a
`JobManager` reporting progress over `WS /ws/session/{sid}`.

### D7 — Client-side instant preview for point operators (optional, Phase 10)

Brightness, contrast, gamma, levels, curves, saturation, temp/tint, invert and
channel gains are all per-pixel LUTs. A small WebGL2 shader can apply them to
the already-loaded proxy texture at 60 fps while the debounced server request is
in flight, then cross-fade to the authoritative render. Keep this strictly an
*accelerator*: the server render is always the source of truth, and the feature
must be removable without touching the pipeline.

### D8 — Cancellation, not debouncing alone

Every preview request carries a monotonically increasing `seq`. The client
debounces 100 ms **and** aborts the in-flight request via `AbortController`; the
server checks a cancellation token between pipeline stages. Dragging a slider
across its full range must never queue 40 renders.

### D9 — Session storage, no database

`ImageStore`: `image_id -> {original float32, proxy, pyramid, exif, created_at}`
in memory, TTL 2 h, LRU-capped by total bytes, backed by a per-session temp
directory. Recipes are client-owned JSON. Nothing needs SQL. (If persistence is
ever wanted: a single SQLite table of `(recipe_id, json, created_at)`.)

### D10 — Upload hardening

Cap request body (default 25 MB), cap decoded megapixels (default 40 MP),
validate the real format by magic bytes rather than extension, reject
decompression bombs before `cv2.imdecode`, strip EXIF on export by default (with
an opt-in to preserve), and apply the EXIF orientation tag at decode so portrait
phone photos are not sideways.

---

## 2. Tech stack

**Backend** — Python 3.11, FastAPI, Uvicorn, Pydantic v2, NumPy,
`opencv-contrib-python`, SciPy, scikit-image, PyWavelets, Pillow (EXIF/ICC),
`anyio`, pytest + pytest-benchmark, ruff + mypy.

**Frontend** — React 18 + TypeScript + Vite, Tailwind CSS, shadcn/ui,
lucide-react, Zustand (pipeline/session state), TanStack Query (server state),
react-dropzone, uPlot (live histograms/waveform — fast enough to redraw per
frame), Recharts (static analysis charts: radar, bars), KaTeX (operator
formulas), react-compare-slider or a custom split view, class-based dark/light
theming over CSS variables.

Interactive canvas overlays (crop, perspective corners, watershed markers,
GrabCut brush, line probe, frequency-domain mask painting) are built on a thin
custom layer over `<canvas>` with pointer events — **not** Konva/Fabric. We need
pixel-exact coordinate mapping through zoom/pan, and a scene-graph library makes
that harder, not easier.

**Note on SURF.** The syllabus lists it, but SURF lives in `opencv_contrib`'s
nonfree `xfeatures2d` module and is absent from PyPI wheels. Options: (a) build
OpenCV with `OPENCV_ENABLE_NONFREE`, (b) substitute AKAZE/KAZE/BRISK, (c)
implement the detector ourselves. **We choose (c) plus (b):** a ~150-line NumPy
integral-image + box-filter approximate-Hessian detector with scale-space
octaves, which is exactly what the paper describes and is far more defensible in
a viva than calling a library, with AKAZE offered alongside for comparison. SIFT
needs no special handling (patent expired; in main OpenCV since 4.4).

---

## 3. Repository layout

```
backend/
  app/
    main.py                 # app factory, CORS, static mount, exception handlers
    config.py               # pydantic-settings
    api/routes/
      images.py  operations.py  pipeline.py  analysis.py
      advisor.py  augment.py  export.py  jobs.py  ws.py
    core/
      registry.py           # OperationSpec, @register, catalogue serialisation
      params.py             # ParamSpec types, coercion, validation, JSON schema
      pipeline.py           # executor, scale-awareness, domain conversion, masks
      cache.py              # byte-bounded LRU + disk spill
      store.py              # ImageStore, sessions, TTL
      imageio.py            # decode/encode, EXIF orient, proxy, pyramid, guards
      colorspace.py         # sRGB<->linear, Lab, HSV, YCbCr, CCT
      masks.py              # brush, gradient, luminosity-range, color-range
      blend.py              # 13 blend modes + opacity
      jobs.py               # JobManager, progress, cancellation tokens
      errors.py
    ops/
      point/ color/ smooth/ sharpen/ freq/ geom/ morph/ edges/ features/
      segment/ augment/ compose/
    analysis/
      exposure.py contrast.py color.py sharpness.py noise.py
      artifacts.py geometry.py texture.py saliency.py spectra.py score.py
    advisor/
      rules.py              # declarative Finding -> Fix rules
      derive.py             # metric -> operator parameters
      recipe.py             # canonical ordering, conflict resolution, capping
      verify.py             # re-measure after apply, score delta, rollback
    reporting/
      report.py             # lab-report generator (markdown + PDF)
  tests/
    test_params.py test_registry.py test_pipeline.py test_scale_invariance.py
    test_ops_golden.py test_analysis.py test_advisor_rules.py bench_ops.py
    fixtures/               # synthetic + real test images, golden outputs

frontend/src/
  App.tsx  routes/
  components/
    ui/                     # shadcn primitives
    layout/                 # LeftRail, SectionNav, Inspector, StatusBar
    canvas/                 # ImageCanvas, CompareSplit, ZoomPan, overlays/
    controls/               # schema-driven renderers (see Section 5)
    charts/                 # Histogram, CDF, Waveform, Vectorscope, Radar, Spectrum
    pipeline/               # OpStack, OpCard, MaskEditor, HistoryPanel
    advisor/                # ScoreCard, FindingList, FindingCard, RecipePreview
  features/studio/ features/advisor/ features/augment/
  lib/ api.ts schema.ts types.ts units.ts format.ts
  store/ session.ts pipeline.ts ui.ts
docs/
  operations-catalogue.md  advisor-spec.md  api.md  theory.md
```

---

## 4. API surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/images` | Upload. Returns `image_id`, dimensions, proxy render id, EXIF summary. |
| `POST` | `/api/images/from-url` | Optional: fetch by URL (SSRF-guarded, Phase 10). |
| `GET` | `/api/operations` | Full catalogue: categories, operators, param schemas, docs. |
| `GET` | `/api/operations/{id}` | One operator + extended docs + formula. |
| `POST` | `/api/operations/{id}/auto` | Compute `auto`-capable params from the image (e.g. Canny thresholds). |
| `POST` | `/api/pipeline/preview` | Body `{image_id, ops[], target, roi?, seq}`. Returns render id + per-op timings + post-stats. |
| `POST` | `/api/pipeline/probe` | Cheap scalar/vector readouts for the current stack top (histogram, metrics) with no new render. |
| `GET` | `/api/render/{render_id}` | Immutable image bytes. |
| `POST` | `/api/analysis` | Body `{image_id, ops?, groups[]}`. Returns metrics JSON + render ids of generated maps. |
| `POST` | `/api/advisor/diagnose` | Returns `{scores, findings[], recipe[]}`. |
| `POST` | `/api/advisor/verify` | Re-measure after a recipe; returns per-dimension deltas. |
| `POST` | `/api/augment/preview` | Sample N variants of a random recipe at thumbnail size. |
| `POST` | `/api/augment/generate` | Job: M images x N variants -> zip + manifest. |
| `POST` | `/api/export` | Job: full-res render with format/quality/bit-depth/resize/metadata options. |
| `GET` | `/api/jobs/{job_id}` | Job status/result. |
| `WS` | `/ws/session/{sid}` | Job progress, cache stats, cancellation acks. |

Errors are uniform: `{error: {code, message, field?, detail?}}`, with codes such
as `IMAGE_TOO_LARGE`, `UNKNOWN_OPERATION`, `PARAM_OUT_OF_RANGE`,
`OP_REQUIRES_GRAYSCALE`, `SECOND_IMAGE_REQUIRED`, `CANCELLED`.

---

## 5. Param type system (`core/params.py`)

The UI is only as good as this list. Each type maps to exactly one React control.

| Type | Fields | React control |
|---|---|---|
| `float` | min, max, step, default, unit, scale_aware, log_scale | Slider + numeric entry |
| `int` | min, max, step, default | Slider (integer snap) |
| `odd_int` | min, max, default | Slider snapping to odd values (kernel sizes) |
| `bool` | default | Switch |
| `enum` | options[{value,label,icon?,help?}] | Select / segmented control |
| `color` | default | Color picker + eyedropper from canvas |
| `point` | normalized xy | Draggable canvas handle |
| `points` | count or min/max, normalized | N draggable handles (perspective, snake seeds) |
| `rect` | normalized | Canvas crop/ROI rectangle |
| `line` | two normalized points | Canvas line probe |
| `range` | min, max, two values | Dual-thumb slider (hue range, luminosity mask) |
| `matrix` | rows, cols, min, max, presets | Editable kernel grid + normalize toggle + presets |
| `curve` | control points, channel selector | Spline editor over a histogram backdrop |
| `image_ref` | — | Second-image slot (histogram matching, color transfer, mixup, registration) |
| `mask_ref` | — | Mask source selector |
| `freq_mask` | — | Paintable frequency-domain mask canvas |
| `lut_file` | accepts `.cube` | File drop |
| `seed` | default | Numeric + dice button |

Shared machinery: `depends_on` (show a param only when another has a given
value), `group` (collapse advanced params), and `auto` (a param computable from
the image — e.g. Canny thresholds from the Otsu level — which gives every such
param an "Auto" button backed by `/api/operations/{id}/auto`).

---

## 6. Phases

Each phase ends in something runnable and demonstrable. Acceptance criteria are
written so they can be verified without judgement calls.

---

### Phase 0 — Foundations

**Build**
- Repo skeleton per Section 3; `pyproject.toml`, Vite app, Tailwind + shadcn
  init, dark/light tokens, lucide icons.
- `main.py` app factory, settings, uniform error envelope, `/api/health`.
- `core/imageio.py`: decode (magic-byte sniff, MP cap, EXIF orientation, ICC
  note), `float32` normalisation, grayscale/alpha handling, proxy + pyramid,
  encode (JPEG/PNG/WebP/TIFF, 8/16-bit).
- `core/store.py` + `core/cache.py` with TTL/LRU/byte accounting.
- `POST /api/images`, `GET /api/render/{id}`.
- Frontend shell: left rail with the two sections; upload dropzone (plus paste
  from clipboard and a sample-image gallery); `ImageCanvas` with zoom/pan/fit;
  before/after split view; status bar showing dimensions, megapixels, proxy
  scale and render timings.

**Acceptance** — Upload a 24 MP phone JPEG, a 16-bit PNG, a grayscale TIFF and a
PNG with alpha; each displays correctly oriented at the right aspect ratio with
pan/zoom working. A 50 MP file is rejected with `IMAGE_TOO_LARGE`. Memory
returns to baseline after TTL eviction.

---

### Phase 1 — Core engine + Tone operators (the MVP ships here)

**Build**
- `core/params.py` (all types from Section 5), `core/registry.py`,
  `core/pipeline.py` (ordered execution, enable/disable, per-op timing,
  scale-aware rescaling, domain conversion, cancellation checks).
- `POST /api/pipeline/preview`, `/probe`, `GET /api/operations`.
- **Schema-driven control renderer**: `ParamControl` dispatches on type;
  `OperatorPanel` composes them with labels, units, reset-to-default,
  double-click-to-default, keyboard arrow nudge, and a "pin to favourites" bit.
- `OpStack` component: add/remove/reorder (drag), toggle, duplicate, rename,
  per-op opacity, collapse, "mute all below".
- Undo/redo over the recipe (command history in Zustand, 100 steps).
- Live histogram + CDF panel (uPlot) with RGB/luma/per-channel modes, log-Y and
  a clipping zebra overlay toggle.
- Operators: brightness, contrast (linear and sigmoid), gamma, exposure (stops),
  log / inverse-log, piecewise contrast stretch, percentile auto-levels, levels
  (black/mid/white plus output range), **curves** (spline, per-channel),
  negative, the threshold family, gray-level slicing, posterize, solarize,
  histogram equalization, AHE, **CLAHE**, histogram matching (to a reference
  image or to a parametric target).
- Reset, download, keyboard shortcuts (`space` = hold before/after, `[`/`]` =
  stack navigation, `ctrl+z`/`ctrl+y`).

**Acceptance** — Every item in `gen_idea.md`'s compulsory table works. A
six-operator stack previews in under 120 ms on a 1600 px proxy with a warm
cache. Dragging a slider end-to-end keeps at most one request in flight.
`test_scale_invariance.py` passes: the same recipe at 1024 px and at native
resolution agrees to SSIM > 0.985 after downscaling the native result.

---

### Phase 2 — Inspect / Understand module

Load-bearing phase: the Advisor is built on these primitives, so they are
implemented as reusable `analysis/*` functions first and *then* surfaced as
Studio views.

**Build**
- `analysis/` modules per `docs/advisor-spec.md`: exposure, contrast, color,
  sharpness, noise, artifacts, geometry, texture, saliency, spectra.
- `POST /api/analysis` with group selection; map-producing analyses return
  render ids (noise map, saliency, dark channel, edge density, entropy map,
  local-contrast map, illumination field, clipping mask).
- Studio "Inspect" views: histogram/CDF; channel splitter (RGB, HSV, Lab,
  YCbCr, CMYK) as a 2x2 grid; bit-plane slicer; intensity line profile; 3D
  intensity surface; waveform monitor; vectorscope; false-color exposure; zone
  map; **DFT magnitude/phase viewer** (log-scaled, fftshifted, frequency ruler);
  radial power spectrum with fitted slope; DCT spectrum; Walsh/Hadamard
  spectrum; GLCM texture stats; metrics panel
  (MSE/PSNR/SSIM/MS-SSIM/UQI/entropy/sharpness vs the original or a reference).
- Pixel inspector: magnifier loupe with the value grid under the cursor in the
  selected color space.
- An **Explain** drawer on every operator and metric: plain-language summary,
  the formula in KaTeX, a kernel/response visualisation, and "what a good value
  looks like".

**Acceptance** — Every metric in the advisor spec is computed and unit-tested
against synthetic images with known ground truth: a flat gray field reports
sigma = 0 noise; a known-sigma Gaussian-noise image is estimated within 15%; a
synthetically blurred edge reports the expected edge width; a tilted synthetic
grid reports the planted angle within 0.2 deg.

---

### Phase 3 — Smoothing / Denoise + Sharpen / Detail

**Build**
- Smoothing: box, Gaussian (separable, sigma or radius), median, weighted
  median, adaptive median (variable window, for salt-and-pepper),
  min/max/midpoint/alpha-trimmed-mean, bilateral, joint/guided filter,
  non-local means, anisotropic diffusion (Perona–Malik, both conduction
  functions), total variation (Chambolle), wavelet denoise
  (BayesShrink/VisuShrink, selectable wavelet and level), adaptive Wiener,
  morphological open/close denoise.
- Sharpen: unsharp mask (radius/amount/threshold), high-boost, Laplacian
  sharpen (4- and 8-neighbour), **custom kernel editor** (3x3..9x9 with
  normalise toggle, symmetry lock, preset library and a live frequency-response
  plot of the entered kernel), Sobel/Prewitt/Roberts/Scharr, emboss,
  local-contrast "clarity" via a bilateral base/detail split, structure-aware
  sharpening gated by a gradient mask, and **deconvolution** (Wiener and
  Richardson–Lucy with Gaussian / motion / disc PSF builders).
- Per-operator opacity and blend exist from Phase 1; add **masks** here —
  luminosity-range and color-range (brush and gradient land in Phase 9) — so
  "sharpen only the midtones" works.

**Acceptance** — Each denoiser improves PSNR against a clean reference on the
noisy fixture set (synthetic noise added to clean plates at sigma = 5/15/30);
the kernel editor's frequency-response plot matches an offline FFT of the same
kernel; unsharp with amount 0 is a numerical no-op (max abs diff < 1e-6).

**Checkpoint: Advisor v0.** With Phases 1–3 done, ship a first vertical slice of
Section 2 covering exposure, contrast, color cast, saturation, sharpness and
noise. Having the second section demonstrable early de-risks Phase 8; the
remaining dimensions (geometry, haze, artifacts, uniformity) plug into the same
rule table later.

---

### Phase 4 — Frequency domain and transforms

**Build**
- 2D DFT with magnitude/phase display, fftshift, log scaling, a padding-strategy
  selector (zero / reflect / `getOptimalDFTSize`) and windowing (Hann/Hamming)
  to suppress edge ringing.
- Frequency-domain filters, each in **Ideal / Butterworth(order n) / Gaussian**
  variants: low-pass, high-pass, band-pass, band-reject, notch (multi-notch with
  draggable positions).
- **Interactive frequency mask painting**: brush directly on the spectrum and
  inverse-transform live. The single most pedagogically effective view in the
  app, and nearly free once the DFT plumbing exists.
- Phase-only and magnitude-only reconstruction (the classic demonstration that
  phase carries structure).
- **Automatic periodic-noise removal**: detect spectral peaks after subtracting
  the radial average, place notches automatically, show before/after. Feeds
  Advisor rule `NOISE_PERIODIC`.
- Homomorphic filtering (log -> Butterworth high-emphasis -> exp) for uneven
  illumination.
- DCT: block and global, coefficient zeroing with a quality/ratio readout — a
  live "this is how JPEG works" demo that ties into the blockiness metric.
- Walsh and Hadamard transforms (fast Hadamard by butterfly on power-of-two
  padded arrays; Walsh = sequency-ordered Hadamard), showing the sequency
  reordering explicitly, with coefficient-truncation reconstruction.
- Haar / Daubechies wavelet decomposition with per-subband gain control and
  reconstruction (shared with the Phase 3 wavelet denoiser).

**Acceptance** — Forward/inverse round-trips to max abs error < 1e-5 for DFT,
DCT, Hadamard, Walsh and DWT; a frequency-domain Gaussian low-pass agrees with a
spatial Gaussian of matched sigma to SSIM > 0.99; an image with a planted
sinusoid is cleaned by auto-notch with the sinusoid's energy reduced by > 30 dB.

---

### Phase 5 — Geometric transforms and Morphology

**Build**
- Geometric: translate, rotate, scale/resize (aspect lock, target by long edge
  or by megapixels), shear, flip/transpose, affine from 3 draggable points,
  **perspective from 4 draggable corners** with a live keystone-corrected
  preview, crop (ratio presets, rule-of-thirds guides), an interpolation
  selector (nearest/bilinear/bicubic/Lanczos4/area) with a side-by-side quality
  comparison view, border-mode selector, auto-deskew (Hough angle histogram),
  lens distortion add/remove (radial k1,k2 plus tangential), vignette add/remove
  (radial polynomial fit), log-polar and polar unwrap, registration by phase
  correlation and by ECC against a second image, and **seam-carving
  content-aware resize** (classical DP over a gradient energy map — cheap to
  implement, visually striking, clearly beyond the syllabus).
- Morphology: a **structuring-element designer** (rect/ellipse/cross/diamond/
  line-at-angle/custom matrix, size, anchor, iterations) shared by every morph
  operator; dilate, erode, open, close, gradient, top-hat, black-hat,
  hit-or-miss, thinning, skeletonize, morphological reconstruction by dilation,
  area opening/closing, fill holes, convex hull, distance transform
  (L1/L2/chessboard — the syllabus' distance metrics), granulometry/pattern
  spectrum chart, grayscale morphology, and a background-estimation-plus-division
  flow for uneven illumination.

**Acceptance** — Rotating by +θ then −θ with Lanczos returns SSIM > 0.97 against
the original interior region; open and close are idempotent to within 1 bit; the
L2 distance transform matches `scipy.ndimage.distance_transform_edt` to 1e-4;
deskew recovers a planted tilt within 0.2 deg.

---

### Phase 6 — Edges, Features and Segmentation (interactive canvas work)

**Build**
- Edges: Canny (hysteresis sliders, aperture/L2-gradient toggle, and an **Auto**
  button deriving thresholds from the Otsu level and from the median heuristic),
  Marr–Hildreth LoG zero-crossings, DoG with a sigma-ratio control, gradient
  magnitude/orientation visualisation (HSV orientation wheel).
- Features: Harris (with a response-surface heatmap), Shi–Tomasi, Hessian /
  determinant-of-Hessian, FAST, SIFT, our **own approximate-Hessian SURF-style
  detector** (Section 2 note), ORB, BRISK, AKAZE, HOG visualisation (cell
  gradient rosettes), LBP plus histogram, a Gabor filter bank (orientation x
  scale grid), Gaussian and Laplacian pyramids with reconstruction and **pyramid
  blending** of two images, keypoint matching between two images (BF/FLANN,
  ratio test, RANSAC homography, inlier count) with a stitching mini-demo,
  template matching (all six TM methods plus the score surface), Hough lines and
  circles with parameter-space visualisation, contour extraction with shape
  descriptors (area, perimeter, circularity, solidity, aspect, extent, Hu
  moments), Fourier descriptors, chain codes.
- Segmentation: global threshold, Otsu, multi-Otsu, Kapur entropy, isodata,
  adaptive mean/Gaussian, Niblack, Sauvola, k-means color quantisation (k, color
  space, palette display), mean-shift, SLIC superpixels (boundary overlay),
  **watershed** (distance-transform markers *and* interactive marker painting),
  seeded region growing (click seeds plus tolerance), split-and-merge quadtree,
  **GrabCut** (rectangle plus foreground/background brush refinement), active
  contours / Chan–Vese, connected components with a stats table and a clickable
  labelled overlay, flood fill, HSV/Lab color-range masking, chroma key.
- Canvas overlay infrastructure: marker brush, seed clicks, rectangle, polygon —
  each with its own undo and correct coordinate mapping under zoom/pan.
- Mask handoff: any segmentation result can be **promoted to a pipeline mask**,
  so "equalize only the background" is two clicks. This is where the
  segmentation section earns its place instead of being a dead-end viewer.

**Acceptance** — Otsu matches `skimage.filters.threshold_otsu` exactly on the
fixture set; SIFT matching between an image and its own rotated/scaled copy
recovers the known homography within 1 px RMS; a promoted watershed mask applied
to a tone operator visibly confines the effect and survives recipe
export/import.

---

### Phase 7 — Augmentation Lab

The third pillar the brief asks for ("augmenting"), built as a *generator*, not
just more filters.

**Build**
- Geometric aug: flip, rotate, random crop, random resized crop, scale jitter,
  translate, shear, perspective jitter, elastic deformation, grid distortion,
  optical distortion.
- Photometric aug: brightness/contrast/saturation/hue jitter, gamma jitter,
  random CLAHE, posterize, solarize, equalize, channel shuffle, random
  grayscale, invert, LUT jitter.
- Degradation aug (the valuable ones, and pure DSP): Gaussian noise, Poisson /
  shot noise, salt-and-pepper, speckle, sensor/ISO simulation (shot plus read
  noise in linear light, then re-encode), **JPEG compression artifacts** at a
  chosen quality, motion blur (angle and length), defocus/disc blur, zoom blur,
  haze/fog synthesis (atmospheric model over a depth proxy), rain and snow
  streak overlays, vignette, chromatic aberration, dropout/cutout, random
  erasing, grid mask, downscale-upscale resolution loss, mixup and cutmix
  against a second image.
- **Recipe composer**: ordered stages, per-stage probability `p`, parameter
  *ranges* rather than fixed values, `OneOf` / `SomeOf` groups, seed control for
  reproducibility.
- **Variant gallery**: an N x M grid of sampled outputs at thumbnail size, with
  regenerate, pin-favourite, and promote-into-the-Studio-stack.
- **Batch generate**: upload or select multiple images, produce K variants each,
  stream progress over WS, download a zip containing the images plus a
  `manifest.json` recording the exact sampled parameters per output — so results
  are reproducible and the manifest is report-ready.
- Preset recipe library: Classification-light, Classification-heavy, Detection
  (geometry-safe), OCR/document, Low-light robustness, Medical-conservative,
  Sensor-simulation.

**Acceptance** — The same seed reproduces a byte-identical variant set; the
manifest replayed through `/api/pipeline/preview` reproduces each image exactly;
a 20-image x 10-variant job completes with live progress and a correct zip.

---

### Phase 8 — Advisor (Section 2, full)

Implements `docs/advisor-spec.md`. The pipeline is **measure -> diagnose ->
derive -> order -> apply -> verify**.

**Build**
1. **Measure** — run the `analysis/*` suite (Phase 2) on the proxy; expensive
   analyses (noise estimation, saliency, Hough) are cached per `image_id`.
2. **Score** — collapse metrics into nine 0–100 dimension scores (Exposure,
   Contrast, Color, Sharpness, Noise, Detail/Texture, Artifacts, Geometry,
   Uniformity) plus one weighted overall "image health". Scoring curves are
   piecewise-linear over documented thresholds held in a single table, not magic
   constants buried in code, so they can be tuned and cited.
3. **Diagnose** — the declarative rule table produces `Finding` objects:
   `{id, dimension, severity, title, explanation, evidence{metric, value,
   expected_range}, confidence}`. Every finding must name the number that
   triggered it — no "looks a bit dark".
4. **Derive** — `advisor/derive.py` turns metrics into concrete operator
   parameters: CLAHE clip limit from RMS contrast, unsharp radius from the
   measured edge width, bilateral sigma from the estimated noise sigma, white
   balance gains from the combined illuminant estimate, notch positions from
   spectral peaks. All formulas are tabulated in the spec.
5. **Order** — canonical pipeline order (geometry -> white balance -> denoise ->
   exposure/tone -> local tone/dehaze -> contrast -> color -> sharpen -> artifact
   cleanup), with conflict resolution: never both global HE and CLAHE; scale the
   sharpen amount down when a denoiser is also in the recipe; cap the recipe at
   eight operators by severity.
6. **Apply** — each finding has a one-click **Fix** appending its operators to
   the Studio stack with the derived parameters visible and editable; the
   Advisor hands off to the Studio rather than owning a second editor. Plus
   **Apply all** and a strength multiplier (Subtle / Balanced / Strong) scaling
   every derived amount by 0.5 / 1.0 / 1.6.
7. **Verify** — re-measure after application and show per-dimension deltas as a
   before/after radar chart. If any dimension *regressed* beyond tolerance,
   surface a warning and a one-click revert. This closed loop is what makes the
   section defensible: "this improved the image" becomes measured rather than
   asserted.

**UI** — `ScoreCard` (overall health ring plus radar); `FindingList` grouped by
severity with evidence chips; each `FindingCard` expandable into the metric and
its expected range, a small chart or map proving it (histogram with the clipped
region highlighted, the noise map, the spectrum with detected peaks circled, the
Hough angle histogram), the proposed operators with parameters, and
Fix / Preview-on-hover / Dismiss actions. Before/after panes stay pinned.

**Acceptance** — On a curated 20-image evaluation set covering underexposure,
overexposure, flat contrast, warm and cool casts, soft focus, motion blur, heavy
noise, salt-and-pepper, haze, tilt, vignetting, uneven illumination, JPEG
blocking and banding (synthetic degradations of clean plates, so ground truth is
known): every planted defect is detected; "Apply all" raises overall health on
at least 18 of 20; no dimension regresses by more than 3 points on any image.
And the inverse criterion — a *good* image must produce few or no findings, so a
false-positive-free run on 5 well-exposed clean plates is part of acceptance.

---

### Phase 9 — Pipeline pro features

**Build**
- **Masks**: brush (size/hardness/flow), linear and radial gradients,
  luminosity range, color range, plus promoted segmentation masks (Phase 6);
  mask invert, feather, blur, levels; overlay display modes (red tint /
  black-white / onion).
- **Blend modes** per op: normal, multiply, screen, overlay, soft light, hard
  light, difference, exclusion, darken, lighten, color-dodge, color-burn,
  luminosity — plus opacity (both stubbed in Phase 1).
- Snapshots, named versions, A/B compare of two snapshots, split slider,
  onion-skin, difference view with gain, synced zoom/pan across panes.
- Recipe export/import as JSON, a shareable recipe string, and a preset library
  (user-saved plus shipped: Low-light Rescue, Noisy Scan Cleanup, Document Scan,
  Astro Stretch, Portrait Soften, Product Shot, Detail Pop, Film Look).
- **Batch apply** a recipe to many images -> zip, with per-image scale-aware
  parameter handling.
- Export dialog: format, quality, chroma subsampling, bit depth, resize,
  output sharpening, metadata keep/strip, ICC note, filename template.

**Acceptance** — A recipe exported, re-imported into a fresh session and applied
to the same image yields a pixel-identical render; a masked operator's effect is
exactly zero where the mask is zero; batch-applying a 10-op recipe to 10 images
completes with correct per-image scale handling.

---

### Phase 10 — Polish, performance, documentation, submission artifacts

**Build**
- Perf: the WebGL point-operator fast path (D7); tile-based rendering of the
  visible viewport at high zoom; a benchmark page showing per-operator
  milliseconds at several resolutions; a cache hit-rate display; lazy module
  imports so SciPy / skimage / PyWavelets do not inflate cold start.
- **Learn mode**: every operator and metric gets its KaTeX formula, a kernel or
  transfer-function plot, a short "what this does / when to use it" note, and a
  reference. This turns the project from a tool into a teaching artifact, which
  is what a lab mini-project is graded on.
- **Auto report generator**: from a session, emit a markdown plus PDF report with
  the input, the applied recipe with each operator's parameters and theory, the
  before/after images, the full metric table with deltas, the Advisor findings
  and the verification radar — the lab write-up builds itself from real runs.
  Includes the "no learned models" statement and its justification.
- Tests: golden-image tests for every operator (tolerance-based, fixtures
  committed), property tests (identity params are no-ops, transforms round-trip),
  a schema-completeness test (every registered operator has docs, a formula, a
  category and bounded params — CI fails otherwise), analysis tests against
  synthetic ground truth, advisor rule tests, API contract tests.
- Accessibility: keyboard reachability for every control, ARIA labels on canvas
  tools, focus-visible rings, respect for `prefers-reduced-motion`, and a
  colour-blind-safe palette for all overlays and charts.
- Empty/error/loading states everywhere; a sample-image gallery so the app is
  explorable with zero uploads; an onboarding tour.
- Packaging: `docker-compose` (api plus web) and a single-container mode where
  FastAPI serves the built static bundle; `make dev`, `make test`, `make bench`;
  a README with screenshots; `docs/api.md` and `docs/theory.md`.

**Acceptance** — Full test suite green; cold start under 3 s; the generated
report for a real session is complete and correct; every registered operator
appears in the UI with documentation (enforced by the schema-completeness test).

---

## 7. Cross-cutting risks and how each is handled

| Risk | Handling |
|---|---|
| Preview latency ruins the "pro editor" feel | Proxy resolution (D3) + prefix cache (D2) + abort-on-new (D8) + optional WebGL point ops (D7). Budget: p95 under 150 ms for a slider move. |
| Export does not match preview | `scale_aware` params (D3) and a dedicated scale-invariance test in Phase 1. |
| Banding from chained operators | `float32` throughout, quantise once (D4). |
| Advisor gives generic, unjustifiable advice | Every finding cites the metric, its value and the expected range; parameters come from documented formulas; the loop closes with measured verification (Phase 8). |
| Advisor fires on already-good images | Explicit false-positive criteria in Phase 8 acceptance; hysteresis bands around thresholds; a confidence field that suppresses low-confidence findings by default. |
| Scope sprawl across ~150 operators | The registry (D1) makes each operator 30–60 lines with no UI work; phases are ordered so each ships something demonstrable. |
| SURF unavailable in wheels | Implement the approximate-Hessian detector ourselves; offer AKAZE alongside (Section 2). |
| Memory blowup from cached float32 arrays | Byte-bounded LRU with disk spill, per-session caps, TTL eviction, upload MP cap (D2, D9, D10). |
| "Is this secretly using AI?" at evaluation | Documented in the generated report; the clustering-vs-learned-model distinction stated; the dependency list contains no ML frameworks. |

---

## 8. Build order summary

```
0  Foundations .................. upload, decode, proxy, canvas, shell
1  Engine + Tone ............... registry, schema UI, op stack, MVP complete
2  Inspect / Understand ........ analysis primitives + all viewers
3  Smooth + Sharpen ............ denoise & detail families   -> Advisor v0 slice
4  Frequency & Transforms ...... DFT/DCT/Walsh/Hadamard/DWT, live spectrum mask
5  Geometric + Morphology ...... warps, SE designer, morph family
6  Edges, Features, Segment .... detectors, matching, interactive segmentation
7  Augmentation Lab ............ random recipes, gallery, batch + manifest
8  Advisor (full) .............. score, diagnose, derive, order, apply, verify
9  Pipeline pro ................ masks, blends, snapshots, presets, batch, export
10 Polish ...................... perf, learn mode, report generator, tests, docker
```

Phases 4, 5, 6 and 7 are mutually independent once 1 and 2 are done, and may be
reordered to suit a demo deadline. Phase 8 needs Phase 2 plus whichever of 3/4/5
supply its fixes — the v0 slice after Phase 3 de-risks it.
