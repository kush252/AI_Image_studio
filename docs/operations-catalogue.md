# Operation Catalogue

Every row becomes one `OperationSpec` in `backend/app/ops/`. Columns:

- **id** — stable string, never renamed (recipes reference it).
- **params** — using the type vocabulary from `PLAN.md` Section 5. `*` marks a
  `scale_aware` parameter that must be rescaled on full-resolution export.
- **notes** — syllabus relevance, implementation hint, or why it is worth having.

`[S]` = on the syllabus. `[+]` = beyond the syllabus, included because it is
better at the job than the syllabus alternative, or because it is what a
professional tool is expected to have.

Operator count by section: Inspect 24, Tone 17, Color 16, Smooth 15, Sharpen 13,
Frequency 14, Geometric 18, Morphology 17, Edges & Features 22, Segmentation 22,
Augment 34, Compose 9. **Total ≈ 221 entries**, of which roughly 150 are
pipeline operators and the rest are views, analyses and composition tools.

---

## 1. Inspect / Understand

These are *views* and *measurements*, not pipeline operators: they read the
current stack output and render a panel, a chart or a map. They do not mutate
the image. Each one also exposes its numbers through `/api/analysis` so the
Advisor can consume them.

| id | Name | Params | Notes |
|---|---|---|---|
| `view.histogram` | Histogram | enum channels (RGB/luma/R/G/B/L/a/b), bool log_y, int bins | Live, uPlot. Clipping zebra overlay. |
| `view.cdf` | Cumulative histogram | enum channel | Explains equalization: the CDF *is* the HE transfer curve. |
| `view.channels` | Channel splitter | enum space (RGB/HSV/HSL/Lab/YCbCr/CMYK) | 2x2 grid, each pane clickable to isolate. |
| `view.bitplanes` | Bit-plane slicing | int plane 0..7, bool all | Shows where the information lives; MSB vs LSB. |
| `view.profile` | Intensity line profile | line, enum channel | Draggable probe line; plot of values along it. |
| `view.surface3d` | 3D intensity surface | int downsample, enum colormap | Intuition for "image as a function f(x,y)". |
| `view.waveform` | Waveform monitor | enum mode (luma/parade/RGB) | Broadcast-style exposure read. |
| `view.vectorscope` | Vectorscope | bool skin_line | Chroma distribution; cast shows as an offset blob. |
| `view.falsecolor` | False-color exposure | — | IRE-style zones; instantly shows clipping. |
| `view.zonemap` | Zone map | int zones=11 | Ansel-Adams zones; feeds the Advisor's tonal-distribution metric. |
| `view.clipmask` | Clipping overlay | float shadow_thr, float highlight_thr | Blue/red overlay for crushed/blown pixels. |
| `view.loupe` | Pixel inspector | point, int grid=9, enum space | Magnifier + numeric value grid under the cursor. |
| `view.dft` | DFT magnitude / phase | enum show (mag/phase/both), bool log, enum window, enum pad | `[S]` 2D-DFT. fftshifted, with a frequency ruler. |
| `view.radial_spectrum` | Radial power spectrum | int bins | Log-log plot + fitted slope -> blur/noise discrimination. |
| `view.dct` | DCT spectrum | enum mode (global/8x8 blocks) | `[+]` Shows energy compaction; JPEG intuition. |
| `view.hadamard` | Hadamard spectrum | int order | `[S]` Fast butterfly on power-of-two padding. |
| `view.walsh` | Walsh spectrum | int order | `[S]` Sequency-ordered Hadamard; show the reordering. |
| `view.dwt` | Wavelet subbands | enum wavelet, int levels | `[+]` LL/LH/HL/HH tiles per level. |
| `view.glcm` | GLCM texture stats | int distance, enum angles, int levels | Contrast, dissimilarity, homogeneity, ASM, energy, correlation. |
| `view.entropy_map` | Local entropy map | odd_int window* | Where the information is dense. |
| `view.saliency` | Saliency map | enum method (spectral_residual/fine_grained) | `[+]` Spectral residual is pure FFT — no model. Used for ROI-weighted metrics. |
| `view.darkchannel` | Dark channel | odd_int patch* | `[+]` Haze estimation (He et al. prior). |
| `view.noisemap` | Noise map | odd_int window* | Per-patch sigma estimate, heatmapped. |
| `view.metrics` | Quality metrics | image_ref reference, enum metrics[] | MSE, RMSE, PSNR, SSIM, MS-SSIM, UQI, NCC, entropy, sharpness, against the original or a reference. |

---

## 2. Tone / Point operations

All are per-pixel LUT-expressible (so all are candidates for the WebGL fast
path, D7).

| id | Name | Params | Notes |
|---|---|---|---|
| `tone.brightness` | Brightness | float offset −1..1 | `s = r + b` |
| `tone.contrast_linear` | Contrast (linear) | float amount −1..1, float pivot | `s = (r − p)·k + p` |
| `tone.contrast_sigmoid` | Contrast (S-curve) | float strength, float pivot | `[+]` Smooth, no hard clipping; what "contrast" means in real editors. |
| `tone.gamma` | Gamma correction | float gamma 0.1..5 (log_scale) | `s = c·r^γ` |
| `tone.exposure` | Exposure (stops) | float stops −5..5 | `[+]` Multiplicative in linear light; `domain: linear`. |
| `tone.log` | Log transform | float c auto | `s = c·log(1+r)`; expands dark range. |
| `tone.inverse_log` | Inverse log | float c | Compresses dark range. |
| `tone.stretch_piecewise` | Piecewise contrast stretch | points (r1,s1),(r2,s2) draggable | Classic textbook stretching with an interactive transfer curve. |
| `tone.autolevels` | Auto levels (percentile) | float low_pct 0..5, float high_pct 95..100, bool per_channel | Per-channel = also removes casts. |
| `tone.levels` | Levels | float black, float mid, float white, float out_lo, float out_hi, enum channel | The professional staple. |
| `tone.curves` | Curves | curve (per channel: RGB/R/G/B/L) | Monotone cubic spline over a histogram backdrop. The single most-used pro control. |
| `tone.negative` | Negative | — | `s = L−1−r` |
| `tone.threshold` | Threshold | float level, enum mode (binary/inv/trunc/tozero/tozero_inv), bool otsu_auto | `[S]` Global thresholding. |
| `tone.gray_slice` | Gray-level slicing | range window, bool preserve_background | Highlights an intensity band. |
| `tone.posterize` | Posterize | int levels 2..64, bool dither | Also the banding demo. |
| `tone.solarize` | Solarize | float threshold | Also an augmentation primitive. |
| `tone.he` | Histogram equalization | enum space (luma/per_channel/Lab-L) | `[S]`-adjacent. Per-channel shifts color — show why luma-only is usually right. |
| `tone.ahe` | Adaptive HE | odd_int tiles* | Shows why CLAHE's clip limit exists (AHE amplifies noise). |
| `tone.clahe` | CLAHE | float clip 0.5..8, int tiles_x*, int tiles_y*, enum space | The workhorse. Advisor's primary contrast fix. |
| `tone.hist_match` | Histogram matching | image_ref target OR enum parametric (uniform/gaussian/rayleigh) | `[+]` Histogram specification; also used for color transfer. |
| `tone.shadows_highlights` | Shadows / Highlights | float shadows, float highlights, float radius*, float midtone_contrast | `[+]` Local tone via a guided-filter base layer. The right fix for simultaneous shadow crush and highlight clipping, which no global curve can solve. |
| `tone.retinex` | Retinex (MSR/MSRCR) | enum variant, float sigmas[3]*, float gain, float offset | `[+]` Strong low-light recovery with color constancy. |
| `tone.bilateral_tonemap` | Durand tone map | float base_contrast, float sigma_space* | `[+]` Classic base/detail HDR compression. |

---

## 3. Color

| id | Name | Params | Notes |
|---|---|---|---|
| `color.wb_grayworld` | White balance (gray world) | float strength | Assumes the scene averages to gray. |
| `color.wb_whitepatch` | White balance (white patch) | float percentile 95..100 | Max-RGB / Retinex illuminant. |
| `color.wb_shades_of_gray` | White balance (shades of gray) | int p=6 | `[+]` Minkowski-norm illuminant; more robust than either above. Advisor combines all three by median. |
| `color.wb_picker` | White balance (picker) | point neutral_pixel | Click a pixel that should be neutral. |
| `color.temp_tint` | Temperature / Tint | float temp −100..100, float tint | Manual override, in Kelvin-equivalent units. |
| `color.channel_gains` | Channel gains | float r, float g, float b | Raw per-channel multipliers. |
| `color.channel_mixer` | Channel mixer | matrix 3x3 | Full linear color transform; includes B&W mixing. |
| `color.hsv_shift` | Hue / Sat / Value | float hue_deg, float sat, float val | |
| `color.vibrance` | Vibrance | float amount | `[+]` Saturation weighted by `(1 − S)`: boosts muted colors, protects already-saturated ones and skin. What the Advisor uses instead of flat saturation. |
| `color.selective` | Selective color | range hue_window, float sat, float light, float hue_shift, float feather | Target one hue family. |
| `color.to_gray` | Color to gray | enum method (luminosity/average/lightness/single_channel/Lab_L/desaturate), float weights | Show that the method matters — a red object vanishes under some weightings. |
| `color.sepia` | Sepia / Duotone | color shadow, color highlight | |
| `color.lut` | Apply 3D LUT | lut_file (.cube), float strength | `[+]` Professional interchange format; trilinear interpolation. |
| `color.transfer` | Color transfer | image_ref source, float strength | `[+]` Reinhard Lab mean/std transfer. Steal the palette of a reference photo — classical, striking, zero ML. |
| `color.gamut_clip` | Gamut clip / soft clip | enum mode, float knee | Fixes channel clipping that causes hue shifts in highlights. |
| `color.ca_correct` | Chromatic aberration fix | float r_scale, float b_scale, bool auto | `[+]` Auto estimates per-channel edge displacement by phase correlation. |

---

## 4. Smoothing / Denoise

| id | Name | Params | Notes |
|---|---|---|---|
| `smooth.box` | Averaging (box) | odd_int k*, bool normalize | Compulsory in the brief. |
| `smooth.gaussian` | Gaussian blur | float sigma*, odd_int k* (auto from sigma) | Separable. `domain: linear` for physical correctness. |
| `smooth.median` | Median | odd_int k* | Compulsory. The salt-and-pepper answer. |
| `smooth.median_weighted` | Weighted median | odd_int k*, matrix weights | |
| `smooth.median_adaptive` | Adaptive median | odd_int k_max* | `[+]` Grows the window until it finds a non-impulse median. Strictly better than plain median at high impulse density, which the Advisor exploits. |
| `smooth.rank` | Min / Max / Midpoint | odd_int k*, enum mode | Order-statistic filters; pepper vs salt removal. |
| `smooth.alpha_trimmed` | Alpha-trimmed mean | odd_int k*, int d | Interpolates between mean and median. |
| `smooth.bilateral` | Bilateral | int d*, float sigma_color, float sigma_space* | Edge-preserving; the first real denoiser. |
| `smooth.guided` | Guided filter | int radius*, float eps, mask_ref guide | `[+]` Faster and halo-free vs bilateral; also the engine behind shadows/highlights and dehaze. |
| `smooth.nlm` | Non-local means | float h, int template*, int search* | `[+]` Best detail retention for Gaussian noise. Slow — run on the proxy, warn on export. |
| `smooth.anisotropic` | Anisotropic diffusion | int iters, float kappa, float gamma, enum conduction | `[+]` Perona–Malik. Smooths within regions, not across edges. |
| `smooth.tv` | Total variation (Chambolle) | float weight, int iters | `[+]` Great on piecewise-smooth content (documents, graphics). |
| `smooth.wavelet` | Wavelet denoise | enum wavelet, int levels, enum method (BayesShrink/VisuShrink), float sigma auto | `[+]` Multi-scale; the Advisor's choice at high sigma. |
| `smooth.wiener` | Adaptive Wiener | odd_int k*, float noise_var auto | Classic optimal linear restoration. |
| `smooth.morph_denoise` | Morphological denoise | enum op (open/close/open-close), SE designer | Binary/impulse cleanup. |
| `smooth.regrain` | Re-grain (dither) | float sigma (8-bit levels), bool monochrome, int seed | `[+]` Adds a calibrated trace of grain. A denoiser removes the noise that was dithering the quantisation steps, uncovering banding that was latent all along; a little grain below the visible threshold restores the dither without restoring the noise. The Advisor appends this to every Gaussian-denoise recipe. |

---

## 5. Sharpen / Detail

| id | Name | Params | Notes |
|---|---|---|---|
| `sharpen.unsharp` | Unsharp mask | float radius*, float amount, float threshold | The standard. Threshold protects flat areas from noise amplification. |
| `sharpen.highboost` | High-boost | float A ≥ 1, odd_int k* | Compulsory in the brief. `s = A·r − blur(r)` |
| `sharpen.laplacian` | Laplacian sharpen | enum kernel (4-/8-neighbour), float amount | Compulsory (high-pass). |
| `sharpen.kernel` | Custom convolution | matrix 3x3..9x9, bool normalize, bool symmetry_lock, enum border, float amount | **Pro centrepiece.** Preset library (sharpen, emboss, edge, motion, identity) plus a live frequency-response plot of whatever the user types. |
| `sharpen.sobel` | Sobel | odd_int k*, enum direction (x/y/mag/angle), bool l2 | `[S]`-adjacent |
| `sharpen.scharr` | Scharr | enum direction | Better rotational symmetry than Sobel at 3x3. |
| `sharpen.prewitt` | Prewitt | enum direction | |
| `sharpen.roberts` | Roberts cross | — | |
| `sharpen.emboss` | Emboss | float angle, float depth | |
| `sharpen.clarity` | Clarity / local contrast | float amount, float radius* | `[+]` Midtone local contrast via a bilateral base layer; "punch" without halos. |
| `sharpen.detail_split` | Base / detail decomposition | float radius*, float base_gain, float detail_gain | `[+]` Exposes the frequency-separation idea directly; lets the user amplify texture while flattening tone. |
| `sharpen.structure_aware` | Masked sharpen | float amount, float edge_thr, float noise_guard | `[+]` Gated by a gradient mask so noise is not sharpened. The Advisor's safe sharpen. |
| `sharpen.deconv` | Deconvolution | enum method (wiener/richardson_lucy), enum psf (gaussian/motion/disc), float psf_size*, float angle, int iters, float noise | `[+]` The *correct* fix for real blur, as opposed to unsharp masking, which only exaggerates edges. Pairs with the Advisor's motion-blur detector, which estimates angle and length from the spectrum. |

---

## 6. Frequency domain and transforms `[S — major syllabus block]`

| id | Name | Params | Notes |
|---|---|---|---|
| `freq.lowpass` | Low-pass filter | enum kind (ideal/butterworth/gaussian), float cutoff*, int order, enum pad, enum window | `[S]` All three kinds, one operator. Ideal shows ringing; that is the lesson. |
| `freq.highpass` | High-pass filter | same | `[S]` |
| `freq.bandpass` | Band-pass | enum kind, float c1*, float c2*, int order | |
| `freq.bandreject` | Band-reject | enum kind, float c1*, float c2*, int order | Periodic-noise removal. |
| `freq.notch` | Notch filter | points notches[], float radius*, enum kind, bool symmetric | Draggable notches on the live spectrum. |
| `freq.notch_auto` | Auto periodic-noise removal | float peak_sensitivity, float radius* | `[+]` Detects spectral peaks after radial-average subtraction, places notches, reports what it removed. Feeds Advisor `NOISE_PERIODIC`. |
| `freq.mask_paint` | Painted frequency mask | freq_mask, bool soft_edges | `[+]` **The best teaching view in the app.** Brush on the spectrum, see the spatial result live. |
| `freq.phase_only` | Phase-only reconstruction | — | Classic demo: structure survives, intensity does not. |
| `freq.mag_only` | Magnitude-only reconstruction | — | The complement; shows phase carries geometry. |
| `freq.homomorphic` | Homomorphic filter | float gamma_l, float gamma_h, float cutoff*, int order | `[+]` log → high-emphasis → exp. The principled fix for uneven illumination. |
| `freq.dct_compress` | DCT compression demo | enum mode (global/8x8), float keep_ratio, bool show_blocks | `[+]` Live quality-vs-ratio readout with PSNR. |
| `freq.hadamard` | Hadamard transform | int order, float keep_ratio | `[S]` Forward/inverse with coefficient truncation. |
| `freq.walsh` | Walsh transform | int order, float keep_ratio | `[S]` Sequency-ordered; show the ordering table. |
| `freq.dwt` | Wavelet transform | enum wavelet, int levels, float subband_gains[] | `[+]` Per-subband gain plus reconstruction; multi-scale sharpening and denoising in one tool. |

Implementation notes: pad to `getOptimalDFTSize`, optionally window to kill
border discontinuity ringing, always `fftshift` for display but never for
filtering math, and keep forward/inverse round-trip error under 1e-5 as a test.

---

## 7. Geometric transforms `[S]`

| id | Name | Params | Notes |
|---|---|---|---|
| `geom.translate` | Translation | float tx, float ty (px or %), enum border | `[S]` |
| `geom.rotate` | Rotation | float angle, point center, bool expand_canvas, enum interp, enum border | `[S]` |
| `geom.scale` | Scaling / Resize | float sx, float sy, bool lock_aspect, enum target_mode (factor/long_edge/megapixels), enum interp | `[S]` `domain: linear` |
| `geom.shear` | Shearing | float shx, float shy, enum border | `[S]` |
| `geom.flip` | Flip / Transpose | enum axis (h/v/both/transpose) | |
| `geom.affine3` | Affine (3-point) | points 3 src, points 3 dst | Draggable handles; shows the 2x3 matrix live. |
| `geom.perspective4` | Perspective (4-point) | points 4 src, points 4 dst, bool auto_rect | `[+]` Keystone correction. Hugely useful on document and architecture photos. |
| `geom.crop` | Crop | rect, enum ratio_preset, bool thirds_guide | |
| `geom.deskew` | Auto deskew | float max_angle, bool auto_crop | `[+]` Hough angle histogram. Advisor `GEOM_TILT` fix. |
| `geom.interp_compare` | Interpolation comparison | float zoom, enum methods[] | View: nearest vs bilinear vs bicubic vs Lanczos side-by-side at high zoom. Pure teaching value. |
| `geom.lens_distort` | Lens distortion | float k1, float k2, float p1, float p2, bool invert | `[+]` Barrel/pincushion both add and remove. |
| `geom.vignette` | Vignette add / remove | float amount, float radius, float feather, bool remove, bool auto_fit | `[+]` Removal fits a radial polynomial to the low-passed luma. Advisor `UNIF_VIGNETTE` fix. |
| `geom.logpolar` | Log-polar transform | point center | `[+]` Rotation/scale become translation — the registration trick. |
| `geom.polar` | Polar unwrap | point center, enum direction | |
| `geom.register_phase` | Register (phase correlation) | image_ref | `[+]` Translation-only alignment, FFT-based. |
| `geom.register_ecc` | Register (ECC) | image_ref, enum motion (translation/euclidean/affine/homography), int iters | `[+]` Intensity-based alignment for the stitching and before/after-pair workflows. |
| `geom.seam_carve` | Content-aware resize | int target_w, int target_h, bool protect_mask | `[+]` Classical DP over a Sobel energy map. Memorable in a demo and entirely non-ML. |
| `geom.pad` | Pad / Canvas | int t,r,b,l, enum border, color fill | |

---

## 8. Morphology `[S]`

Every entry shares the **structuring-element designer**: `enum shape`
(rect/ellipse/cross/diamond/line/custom), `int size*`, `float angle`,
`point anchor`, `int iterations`, `matrix custom_se`.

| id | Name | Params | Notes |
|---|---|---|---|
| `morph.dilate` | Dilation | SE | `[S]` |
| `morph.erode` | Erosion | SE | `[S]` |
| `morph.open` | Opening | SE | `[S]` Removes small bright specks. |
| `morph.close` | Closing | SE | `[S]` Fills small dark gaps. |
| `morph.gradient` | Morphological gradient | SE | Edge detector by dilation − erosion. |
| `morph.tophat` | Top-hat (white) | SE large* | Extracts bright detail; the uneven-illumination workhorse. |
| `morph.blackhat` | Black-hat | SE large* | Extracts dark detail — text on an uneven background. |
| `morph.hitmiss` | Hit-or-miss | matrix se_fg, matrix se_bg | Shape matching on binary images. |
| `morph.thin` | Thinning | int iters | |
| `morph.skeleton` | Skeletonize | enum method (zhang_suen/lee/medial_axis) | |
| `morph.reconstruct` | Reconstruction by dilation | mask_ref marker, SE | `[+]` Geodesic reconstruction — removes objects without deforming the ones that remain. Much better than opening when shape must be preserved. |
| `morph.area_open` | Area opening / closing | int min_area, enum connectivity | `[+]` Size-based rather than shape-based removal. |
| `morph.fill_holes` | Fill holes | enum connectivity | |
| `morph.convex_hull` | Convex hull | bool per_object | |
| `morph.distance` | Distance transform | enum metric (L1/L2/chessboard) `[S]`, enum mask_size, bool normalize | `[S]` The syllabus' distance metrics; also supplies watershed markers. |
| `morph.granulometry` | Granulometry | enum shape, int max_size*, int step | `[+]` Pattern spectrum chart: object-size distribution. Real analysis, not just a filter. |
| `morph.bg_divide` | Background estimate + divide | int se_size*, enum method (open/close/median) | `[+]` Flattens uneven illumination. Advisor `UNIF_ILLUM` fix, alternative to homomorphic. |

---

## 9. Edges, Features and Description `[S]`

| id | Name | Params | Notes |
|---|---|---|---|
| `edge.canny` | Canny | float t1 (auto), float t2 (auto), odd_int aperture, bool l2_gradient, bool auto_otsu | `[S]` Auto modes: Otsu-derived and median-based (0.66σ/1.33σ). |
| `edge.log` | Marr–Hildreth / LoG | float sigma*, bool zero_crossings | `[S]`-adjacent |
| `edge.dog` | Difference of Gaussians | float sigma1*, float ratio, bool normalize | `[S]` |
| `edge.gradient_vis` | Gradient magnitude / orientation | enum operator, bool hsv_wheel | Orientation as hue, magnitude as value. |
| `feat.harris` | Harris corners | int block*, odd_int ksize, float k, float thresh, bool show_response | `[S]` Response heatmap alongside the keypoints. |
| `feat.shitomasi` | Shi–Tomasi | int max_corners, float quality, float min_dist* | `[S]`-adjacent |
| `feat.hessian` | Hessian / det-of-Hessian | float sigma*, float thresh | `[S]` |
| `feat.fast` | FAST | int threshold, bool nms | |
| `feat.sift` | SIFT | int n_features, int octave_layers, float contrast_thr, float edge_thr, float sigma | `[S]` In main OpenCV since 4.4. Draw keypoints with size and orientation. |
| `feat.surf_approx` | SURF (our implementation) | float hessian_thr, int octaves, int layers | `[S]` Hand-written: integral image + box-filter Hessian approximation + scale-space NMS. See `PLAN.md` Section 2 for why. |
| `feat.akaze` | AKAZE | float thresh, int octaves | `[+]` Modern, free, available in wheels; offered as the SURF comparison. |
| `feat.orb` | ORB | int n_features, float scale_factor, int levels | |
| `feat.brisk` | BRISK | int thresh, int octaves | |
| `feat.hog` | HOG | int cell*, int block, int bins, bool signed | `[S]` Visualise as per-cell gradient rosettes plus the descriptor histogram. |
| `feat.lbp` | LBP | int radius*, int points, enum method | `[+]` Texture descriptor plus histogram. |
| `feat.gabor` | Gabor filter bank | float wavelength*, int orientations, float sigma*, float gamma, float psi | `[+]` Orientation x scale response grid; the biologically-motivated texture tool. |
| `feat.pyramid_gauss` | Gaussian pyramid | int levels | `[S]` Scale-space analysis. |
| `feat.pyramid_laplacian` | Laplacian pyramid | int levels, bool reconstruct | `[S]` Show that reconstruction is lossless. |
| `feat.pyramid_blend` | Pyramid blending | image_ref, mask_ref, int levels | `[+]` Seamless two-image blend. The classic payoff of pyramids. |
| `feat.match` | Keypoint matching | image_ref, enum detector, enum matcher (BF/FLANN), float ratio, bool ransac, bool draw_inliers | `[+]` Reports inlier count and the recovered homography; drives the stitch demo. |
| `feat.stitch` | Two-image stitch | image_ref, enum blend | `[+]` Payoff demo for matching plus homography plus blending. |
| `feat.template` | Template matching | image_ref template, enum method (all 6), bool show_surface | |
| `feat.hough_lines` | Hough lines | float rho*, float theta, int threshold, enum mode (standard/probabilistic), bool show_accumulator | `[S]`-adjacent. Accumulator view makes parameter space concrete. |
| `feat.hough_circles` | Hough circles | float dp, float min_dist*, int p1, int p2, int min_r*, int max_r* | |
| `feat.contours` | Contours + shape descriptors | enum mode, enum method, float min_area, bool table | Area, perimeter, circularity, solidity, extent, aspect, Hu moments — as a sortable table. |
| `feat.fourier_desc` | Fourier descriptors | int n_coeffs, bool reconstruct | `[+]` Shape from a handful of coefficients. |
| `feat.chaincode` | Chain code | enum connectivity (4/8) | Freeman codes plus the shape number. |

---

## 10. Segmentation `[S]`

| id | Name | Params | Notes |
|---|---|---|---|
| `seg.threshold_global` | Global threshold | float level, bool invert | `[S]` |
| `seg.otsu` | Otsu | bool invert, bool show_criterion | `[S]` Plot the between-class variance curve and mark the chosen level. |
| `seg.multiotsu` | Multi-Otsu | int classes 2..5 | `[+]` |
| `seg.kapur` | Entropy threshold (Kapur) | — | `[+]` |
| `seg.isodata` | Isodata / iterative | float tol | Textbook iterative selection. |
| `seg.adaptive` | Adaptive threshold | enum method (mean/gaussian), odd_int block*, float C | Uneven illumination. |
| `seg.niblack` | Niblack | odd_int window*, float k | `[+]` |
| `seg.sauvola` | Sauvola | odd_int window*, float k, float r | `[+]` The document-binarisation standard; markedly better than Otsu on scans. |
| `seg.kmeans` | k-means color quantization | int k, enum space (RGB/Lab/HSV), int iters, seed, bool show_palette | Clustering, not a model (see `PLAN.md` Section 0). Lab space gives perceptually better clusters. |
| `seg.meanshift` | Mean-shift segmentation | float spatial_radius*, float color_radius, int min_density | `[+]` |
| `seg.slic` | SLIC superpixels | int n_segments, float compactness, bool boundaries | `[+]` |
| `seg.watershed` | Watershed | enum markers (distance/manual/gradient), float dist_thresh, mask_ref manual_markers | `[S]` Both automatic (distance transform) and interactive marker painting. |
| `seg.region_grow` | Seeded region growing | points seeds, float tolerance, enum connectivity | Click to seed. |
| `seg.split_merge` | Split and merge | float homogeneity, int min_block | `[+]` Quadtree; visualise the tree. |
| `seg.grabcut` | GrabCut | rect, mask_ref brush, int iters | `[S]` (graph technique). Iterative graph cut with brush refinement. |
| `seg.graphcut_simple` | Min-cut / max-flow | mask_ref fg_bg, float lambda | `[S]` The explicit graph-technique demo, with the energy terms shown. |
| `seg.chanvese` | Chan–Vese level sets | int iters, float mu, float lambda1, float lambda2 | `[+]` |
| `seg.snakes` | Active contours | points init, float alpha, float beta, float gamma, int iters | `[+]` |
| `seg.connected` | Connected components | enum connectivity, float min_area, bool stats_table | Labelled overlay with a clickable stats table. |
| `seg.floodfill` | Flood fill | point seed, float lo_diff, float hi_diff, bool fixed_range | |
| `seg.color_range` | Color range mask | range hue, range sat, range val, enum space, float feather | HSV/Lab masking; the practical "select the sky" tool. |
| `seg.chroma_key` | Chroma key | color key, float tolerance, float spill | |

Every segmentation operator can **promote its result to a pipeline mask**
(`PLAN.md` Phase 6), which is what connects this section to the rest of the app.

---

## 11. Augmentation Lab

Each entry supports **ranges** rather than fixed values, a per-stage probability
`p`, and a seed. `OneOf` and `SomeOf` group containers compose them.

**Geometric** — `aug.flip`, `aug.rotate`, `aug.random_crop`,
`aug.random_resized_crop`, `aug.scale_jitter`, `aug.translate`, `aug.shear`,
`aug.perspective_jitter`, `aug.elastic`, `aug.grid_distort`,
`aug.optical_distort`.

**Photometric** — `aug.brightness_jitter`, `aug.contrast_jitter`,
`aug.saturation_jitter`, `aug.hue_jitter`, `aug.gamma_jitter`,
`aug.random_clahe`, `aug.posterize`, `aug.solarize`, `aug.equalize`,
`aug.channel_shuffle`, `aug.to_gray`, `aug.invert`, `aug.lut_jitter`.

**Degradation** `[+ — the valuable half]` — `aug.gaussian_noise`,
`aug.poisson_noise`, `aug.salt_pepper`, `aug.speckle`, `aug.iso_simulation`
(shot + read noise in linear light, then re-encoded — physically faithful),
`aug.jpeg_artifacts` (quality range; real encode/decode round-trip),
`aug.motion_blur` (angle, length), `aug.defocus_blur` (disc PSF),
`aug.zoom_blur`, `aug.haze` (atmospheric model with a depth proxy),
`aug.rain`, `aug.snow`, `aug.vignette`, `aug.chromatic_aberration`,
`aug.downscale_upscale`, `aug.cutout`, `aug.random_erase`, `aug.grid_mask`,
`aug.mixup` (image_ref, alpha), `aug.cutmix` (image_ref, rect).

**Infrastructure** — recipe composer, variant gallery (N x M thumbnails,
regenerate, pin, promote-to-Studio), batch generate (K variants x M images -> zip
plus `manifest.json` of the exact sampled parameters), preset library
(Classification-light/heavy, Detection geometry-safe, OCR/document, Low-light
robustness, Medical-conservative, Sensor-simulation).

---

## 12. Compose / Pipeline tools

| id | Name | Params | Notes |
|---|---|---|---|
| `compose.blend` | Blend with image | image_ref, enum mode (13 modes), float opacity | |
| `compose.difference` | Difference view | image_ref, float gain | Amplified diff — makes subtle changes visible. |
| `compose.mask_brush` | Brush mask | brush strokes, float size*, float hardness, float flow | |
| `compose.mask_gradient` | Gradient mask | enum type (linear/radial), points, float feather | |
| `compose.mask_luminosity` | Luminosity mask | range, float feather | "Only the midtones." |
| `compose.mask_color` | Color-range mask | range hue/sat/val, float feather | |
| `compose.mask_ops` | Mask operations | enum op (invert/blur/feather/levels/combine), params | |
| `compose.snapshot` | Snapshot | string name | A/B compare anchor. |
| `compose.roi` | Limit to ROI | rect, float feather | Apply everything below only inside a region. |

---

## Implementation conventions

1. One file per operator family, one `@register` per operator. Keep operator
   functions pure: `np.ndarray -> np.ndarray`, no I/O, no globals.
2. Always `float32` in `[0,1]`; never quantise mid-pipeline (`PLAN.md` D4).
3. Mark every length/size/sigma/cutoff parameter `scale_aware` (D3). When in
   doubt, mark it — the scale-invariance test will catch an over-marking faster
   than a user will catch an under-marking.
4. Declare `domain` honestly: blur, resize, exposure and anything modelling
   light transport are `linear`.
5. Declare `requires: {channels: 1|3|any, binary: bool, second_image: bool}` so
   the UI can disable an operator with a clear reason instead of erroring.
6. Every operator needs `summary`, `formula_latex`, `when_to_use` and a
   reference. The schema-completeness test (Phase 10) fails the build otherwise —
   this is what keeps Learn mode honest as the catalogue grows.
7. Prefer OpenCV for speed, scikit-image where its implementation is better
   documented or more correct (TV, Chan–Vese, SLIC, Sauvola, multi-Otsu,
   skeletonize), SciPy for `ndimage` and signal work, PyWavelets for DWT. Where
   a textbook algorithm is short and the point is to show we understand it
   (Walsh/Hadamard ordering, SURF-style Hessian, split-and-merge, seam carving,
   region growing, chain codes, Fourier descriptors, granulometry), write it
   ourselves in NumPy and say so in the docstring.
