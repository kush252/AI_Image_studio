# IPCV Platform — Concept Reference

A reference for writing up the project: what was built, which image-processing
concepts it uses, and where each one appears.

**Hard constraint, held throughout:** no learned model, no neural network, no
pretrained weights. Every result comes from classical image processing. The
API advertises this at `/api/health` as `"uses_learned_models": false`.

---

## 1. What the system is

Two sections over one processing core.

| | Studio | Advisor |
|---|---|---|
| Who drives | The user | The system |
| Purpose | Apply any operator manually, inspect the result | Measure the image, name what is wrong, offer one-click fixes |
| Scale | 239 operators in 13 families | 95 metrics → 9 health scores → 33 diagnostic rules |

**Stack.** FastAPI + OpenCV/NumPy/SciPy/scikit-image/PyWavelets backend;
React + TypeScript + Vite + Tailwind + Radix UI frontend.

**Totals:** 239 operators (80 on the syllabus, 159 beyond it), 13 families,
33 advisor rules, 477 passing tests.

---

## 2. Processing core (shared by both sections)

| Concept | Where used | Why |
|---|---|---|
| **float32 pipeline** | `core/pipeline.py` | All operators work in `[0,1]` float, not uint8, so repeated operations do not accumulate quantisation error |
| **Linear vs gamma domain** | `core/colorspace.py` | Operators declare `domain="linear"`; the pipeline applies EOTF before and OETF after. Blurring and exposure are only physically correct in linear light |
| **Colour spaces** | `core/colorspace.py` | RGB, YCbCr, HSV, CIE L\*a\*b\* — chroma-only denoising, cast measurement in `a*b*`, luma-only sharpening |
| **Non-destructive op stack** | `core/pipeline.py` | Operations are a list, not pixels. Any step can be edited or removed |
| **Prefix caching** | `core/cache.py` | Key = `sha1(image_id │ target │ ops[0..i])`. Editing step 5 of 8 reuses steps 1–4 |
| **Proxy resolution** | `core/imageio.py`, `core/params.py` | Preview at ≤1600 px; `scale_aware` parameters (kernel sizes) scale with resolution and `scale_inverse` ones (normalised frequencies) scale against it, so full-resolution export matches the preview |
| **Declarative registry** | `core/registry.py` | Each operator declares its parameters; the UI is generated from that schema, so no operator needs hand-written UI |

---

## 3. Studio — concepts by family

`[S]` = syllabus topic. Counts are *operators*, not concepts.

### 3.1 Inspect — 21 ops (5 `[S]`)
Measurement and visualisation, no pixels changed.

`[S]` **Histogram**, **cumulative histogram (CDF)**, **bit-plane slicing**,
**DFT magnitude/phase**, **DCT spectrum**.
Beyond syllabus: waveform monitor, vectorscope, zone map, 3-D intensity
surface, radial power spectrum, **GLCM texture statistics**, **local entropy
map**, **saliency map** (spectral residual), **dark channel**, noise map,
false-colour exposure, clipping overlay, intensity line profile, quality
metrics, channel splitter, pixel inspector.

### 3.2 Tone — 23 ops (6 `[S]`)
Point operations on intensity.

`[S]` **Negative**, **log transform**, **inverse log**, **power-law (gamma)**,
**piecewise contrast stretch**, **gray-level slicing**, **histogram
equalization**, **thresholding**.
Beyond syllabus: **CLAHE**, adaptive HE, **histogram matching
(specification)**, **Retinex (MSR/MSRCR)**, **Durand tone mapping**,
shadows/highlights, curves, levels, auto-levels (percentile), exposure in
stops, S-curve contrast, posterize, solarize.

### 3.3 Colour — 16 ops
`[S]` none directly, but built on the colour-space theory above.

**Illuminant estimation / white balance** — gray world, white patch,
**shades-of-gray (Minkowski norm)**, picker. Plus channel gains, channel
mixer, temperature/tint, HSV shift, vibrance (saturation-weighted),
selective colour, 3-D LUT, colour transfer (Reinhard), gamut soft-clip,
**chromatic aberration correction**, colour→gray, sepia/duotone.

### 3.4 Smoothing / Denoising — 16 ops (7 `[S]`)
`[S]` **Averaging (box)**, **Gaussian blur**, **median**, **weighted
median**, **min/max/midpoint**, **alpha-trimmed mean**, **adaptive median**.
Beyond syllabus: **bilateral**, **non-local means**, **anisotropic diffusion
(Perona–Malik)**, **total variation (Chambolle)**, **wavelet denoise
(BayesShrink/VisuShrink)**, **adaptive Wiener**, **guided filter**,
morphological denoise, **re-grain / dither**.

> **Re-grain** was added because of a measured effect: a denoiser removes the
> noise that was *dithering* the quantisation steps, uncovering banding that
> was latent in the image all along. A trace of grain restores the dither.

### 3.5 Sharpening / Detail — 13 ops (7 `[S]`)
`[S]` **Laplacian sharpening**, **unsharp mask**, **high-boost**, **Sobel**,
**Prewitt**, **Roberts cross**, **custom convolution mask**.
Beyond syllabus: **Scharr**, **deconvolution (Wiener and
Richardson–Lucy)**, structure-aware (gradient-masked) sharpening, clarity /
local contrast, base-detail decomposition, emboss.

### 3.6 Frequency domain — 13 ops (9 `[S]`)
`[S]` **DFT/FFT**, **ideal / Butterworth / Gaussian low-pass**, **high-pass**,
**band-pass / band-reject**, **notch filter**, **homomorphic filter**,
**Walsh transform**, **Hadamard transform**, **DCT**.
Beyond syllabus: **discrete wavelet transform**, automatic periodic-noise
removal (peak detection + notching), painted frequency masks, and the
**phase-only / magnitude-only reconstruction** demonstration — which shows
that phase carries the structure.

### 3.7 Geometric — 18 ops (5 `[S]`)
`[S]` **Translation**, **rotation**, **scaling**, **shearing**,
**interpolation** (nearest / bilinear / bicubic / Lanczos comparison).
Beyond syllabus: **affine (3-point)**, **perspective (4-point homography)**,
**log-polar** and **polar unwrap**, lens-distortion correction, **auto
deskew**, **registration** by phase correlation and by ECC,
**content-aware resize (seam carving)**, vignette add/remove, crop, pad, flip.

### 3.8 Morphology — 18 ops (14 `[S]`)
The most syllabus-dense family.

`[S]` **Erosion**, **dilation**, **opening**, **closing**, **hit-or-miss**,
**morphological gradient** (boundary extraction), **top-hat**, **black-hat**,
**thinning**, **skeletonization**, **convex hull**, **fill holes**,
**grayscale morphology**.
Beyond syllabus: **reconstruction by dilation**, **distance transform**,
**granulometry (pattern spectrum)**, area opening/closing, background
estimation and division.

### 3.9 Edges — 4 ops (3 `[S]`)
`[S]` **Canny** (full pipeline: smoothing → gradient → non-maximum
suppression → hysteresis), **Marr–Hildreth / LoG zero crossings**, **gradient
magnitude and orientation**. Beyond: **Difference of Gaussians**.

### 3.10 Features — 23 ops (12 `[S]`)
`[S]` **Harris corners**, **Shi–Tomasi**, **SIFT**, **SURF**, **Hough lines**,
**Hough circles**, **chain codes**, **Fourier descriptors**, **contours and
shape descriptors**, **template matching**, **Gaussian pyramid**, **Laplacian
pyramid**.
Beyond syllabus: **FAST**, **BRISK**, **ORB**, **AKAZE**,
**Hessian / determinant-of-Hessian**, **HOG**, **LBP**, **Gabor filter
bank**, keypoint matching, pyramid blending, two-image stitching.

> **SURF is hand-implemented** (integral images + box-filter approximate
> Hessian), not called from a library — it is a designed operator, not a
> trained one.

### 3.11 Segmentation — 22 ops (12 `[S]`)
`[S]` **Global thresholding**, **Otsu**, **Multi-Otsu**, **isodata
(iterative)**, **adaptive/local thresholding**, **Niblack**, **Sauvola**,
**entropy threshold (Kapur)**, **region growing**, **split-and-merge
(quadtree)**, **watershed**, **connected components**.
Beyond syllabus: **GrabCut**, **Chan–Vese level sets**, **active contours
(snakes)**, **SLIC superpixels**, **mean-shift**, **min-cut/max-flow**,
**k-means colour quantization**, flood fill, colour-range and chroma-key
masks.

### 3.12 Augmentation — 42 ops
Dataset augmentation and **degradation modelling** — the forward models whose
inverses the Advisor applies.

Photometric: brightness/contrast/gamma/hue/saturation jitter, equalize,
random CLAHE, posterize, solarize, invert, grayscale, channel shuffle.
Geometric: flip, rotate, shear, translate, scale, zoom, random crop,
**elastic deformation**, **grid distortion**, **optical distortion**,
perspective jitter.
Noise and degradation: **Gaussian**, **salt-and-pepper**, **Poisson/shot**,
**speckle (multiplicative)**, **sensor/ISO simulation**, **JPEG artifacts**,
**motion blur**, **defocus blur**, **zoom blur**, resolution loss.
Weather/atmosphere: **haze/fog** (atmospheric scattering model), rain, snow,
vignette, chromatic aberration.
Occlusion/mixing: cutout, grid mask, random erasing, **Mixup**, **CutMix**.

### 3.13 Compose — 10 ops
Masking and blending so any operator can be applied **locally**:
**luminosity masks**, gradient masks, colour-range masks, rectangle and brush
masks, mask set operations, ROI limiting, **blend modes**, snapshots,
difference view.

---

## 4. Advisor — measure, diagnose, fix, verify

A seven-stage pipeline. Nothing is suggested that has not been measured.

```
measure → score → diagnose → derive → order → apply → verify
```

### 4.1 Measure — 95 metrics across 9 dimensions

| Dimension | Representative metrics | Concepts used |
|---|---|---|
| **Exposure** (15) | `mean_y`, percentiles, `clip_shadow/highlight`, `range_usage`, zone histogram, **regional** tile means and spread | Histogram statistics, Ansel Adams zone system, linear-light mean |
| **Contrast** (7) | `rms_contrast`, **Michelson**, **entropy**, `local_contrast`, occupancy | Histogram entropy, local vs global contrast |
| **Colour** (15) | `cast_magnitude`, **CCT estimate**, `a*b*` offset, `sat_mean`, `gray_frac` | Illuminant estimation, CIE L\*a\*b\*, correlated colour temperature |
| **Sharpness** (14) | **Variance of Laplacian**, **Tenengrad**, `edge_width`, **regional** edge width, `hf_ratio`, **spectrum slope**, **anisotropy**, **motion length/angle**, `ringing` | Edge-spread function, power-spectrum analysis, **cepstral motion estimation**, **ISO 12233 overshoot** |
| **Noise** (15) | `sigma_luma`, `sigma_chroma`, `impulse_frac`, `impulse_extremity`, periodic peaks, `snr_db` | **Immerkaer kernel** estimation on flat tiles, median-residual outliers, spectral peak isolation |
| **Artifacts** (6) | `blockiness`, `banding_score`, `hist_comb`, `jpeg_quality_est` | 8×8 block-boundary energy, histogram combing / empty-bin runs |
| **Geometry** (4) | `tilt_angle`, `ca_shift_px`, `border_frac` | **Hough-line** dominant angle, per-channel edge registration |
| **Uniformity** (13) | `vignette_falloff`, `illum_nonuniformity`, **`dark_channel_mean`**, airlight, transmission | **Dark-channel prior**, radial and planar illumination fitting |
| **Texture** (6) | `edge_density`, `detail_loss`, **GLCM**, **LBP uniformity**, `graphic_frac` | Haralick texture features, local binary patterns |

### 4.2 Score — 9 dimension scores → one health figure

Each metric is mapped through a **piecewise-linear** good→bad curve to 0–100,
combined per dimension by weight, then weighted into an overall score:

```
exposure .18 │ contrast .15 │ sharpness .15 │ noise .14 │ colour .13
uniformity .09 │ artifacts .07 │ detail .05 │ geometry .04
```

Capped at `min(dimension) + 25`, so one severe defect cannot be averaged away.

### 4.3 Diagnose — 33 declarative rules

| Group | Rules |
|---|---|
| Exposure | `EXP_UNDER`, `EXP_OVER`, `EXP_DUAL_CLIP`, `EXP_SHADOW_CLIP`, `EXP_REGIONAL`, `EXP_NARROW` |
| Contrast | `CON_FLAT`, `CON_LOCAL_FLAT`, `CON_LOW_ENTROPY`, `CON_OVER` |
| Colour | `COL_CAST`, `COL_DULL`, `COL_OVERSAT`, `COL_CHANNEL_CLIP` |
| Sharpness | `SHP_MOTION`, `SHP_SOFT`, `SHP_REGIONAL`, `SHP_NOISE_LIMITED`, `SHP_OVER` |
| Noise | `NSE_GAUSS`, `NSE_IMPULSE`, `NSE_CHROMA`, `NSE_PERIODIC`, `NSE_SHADOW` |
| Artifacts | `ART_BLOCK`, `ART_BAND`, `ART_PRESTRETCHED` |
| Geometry | `GEO_TILT`, `GEO_CA` |
| Uniformity | `UNI_HAZE`, `UNI_VIGNETTE`, `UNI_ILLUM` |
| Detail | `DET_PLASTIC` |

Each rule carries a predicate, an explanation that **must** interpolate the
triggering measurement, the evidence, and a confidence value.

Three context modes change the rules:
- **Document mode** — high edge density + bimodal histogram → Sauvola and
  background division replace CLAHE and homomorphic filtering.
- **Graphic mode** — a chart or screenshot has deliberate pure blacks and few
  distinct levels, so banding, blocking, haze and clipping rules are suppressed.
- **Monochrome** — colour rules are suppressed.

### 4.4 Derive — parameters computed, never guessed

Every parameter is computed from the measurement and ships with the arithmetic
that produced it. Examples:

| Finding | Derivation |
|---|---|
| Underexposure | `EV = log2(0.18 / mean_y_linear)`, clamped |
| Colour cast | Channel gains from the **shades-of-gray illuminant** estimate |
| Motion blur | PSF length and angle from the **cepstral** estimate → Wiener deconvolution |
| Gaussian noise | **Non-local means** `h = 1.0·σ`, then re-grain at `0.22·σ` |
| Impulse noise | Adaptive median, `k_max = 3 + 2·ceil(40·impulse_frac)` |
| Periodic noise | Notch filter at the detected spectral peaks |
| Haze | **Dark-channel prior** dehaze with guided-filter transmission refinement |

### 4.5 Order — canonical pipeline order

Fixes are sorted into stages, because order changes the result: **geometry →
white balance → denoise → exposure → local tone → colour → sharpen**. Denoise
must precede sharpening, or the sharpener amplifies the noise.

**Root-cause suppression:** haze alone makes contrast, exposure, saturation
and illumination all look wrong. When `UNI_HAZE` fires, the downstream fixes
are withdrawn rather than compounded.

### 4.6 Verify — the closed loop

The recipe is applied and the image re-measured *before anything is offered*.

- **Expected trade-offs** are separated from faults: denoising necessarily
  costs high-frequency detail; lifting shadows necessarily reveals the noise
  that was hiding in them. These are the operation working, not failing.
- **Self-refinement** backs off — dropping the worst operator, or halving the
  strength — for up to three rounds, keeping the best measured variant.
- **Do-no-harm guard:** a recipe is withdrawn unless the dimensions it targets
  improve without unexpected collateral damage. An image already in good shape
  may spend no health at all.

---

## 5. Validation

| | |
|---|---|
| Backend tests | **477 passing** |
| Operator panels rendered headlessly | **239 / 239**, 0 failures |
| Synthetic evaluation set | **20 / 20** planted defects detected, 0 false findings on a clean plate |
| Real photographs | **0 of 6 made worse** |

**A result worth reporting.** The advisor was first calibrated on synthetic
degradations and scored 20/20. Run against six real photographs it detected
**1 of 6** and made **3 worse** — the calibration had fitted the generator, not
the world. Recalibrating on real images required regional metrics (a frame-wide
average hides a blurred subject or an unevenly lit half), a noise-independent
impulse measure, and the do-no-harm guard.

Two of those six photographs carry filenames describing their *content*, not a
defect: one is a deliberate panning shot (sharp subject, streaked background,
97.8/100) and one is a screenshot of a well-exposed photo. **Declining to act
on them is the correct behaviour** — and is itself a result: an automatic
enhancer must be able to say "nothing is wrong here."

---

## 6. Where to look

| File | Contents |
|---|---|
| `docs/operations-catalogue.md` | Every operator, its parameters and notes |
| `docs/advisor-spec.md` | Metric formulas, scoring curves, rule table, derivations |
| `docs/api.md` | REST endpoints |
| `backend/app/ops/` | Operator implementations, one module per family |
| `backend/app/analysis/metrics.py` | All 95 measurements |
| `backend/app/advisor/` | `rules.py`, `derive.py`, `recipe.py`, `verify.py` |
| `samples/` | The six real test photographs |
