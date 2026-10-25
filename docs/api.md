# HTTP API

Base URL `http://127.0.0.1:8000`. All bodies are JSON unless noted.
Interactive docs at `/docs` (OpenAPI) once the server is running.

Errors are uniform:

```json
{ "error": { "code": "PARAM_OUT_OF_RANGE", "message": "...", "field": "gamma" } }
```

| Code | Status | Meaning |
|---|---|---|
| `IMAGE_TOO_LARGE` | 413 | Over the byte or megapixel cap |
| `UNSUPPORTED_FORMAT` | 415 | Magic bytes did not match a known format |
| `DECODE_FAILED` | 400 | Corrupt file |
| `IMAGE_NOT_FOUND` | 404 | Expired from the store (2 h TTL) |
| `RENDER_NOT_FOUND` | 404 | Render evicted from the LRU |
| `UNKNOWN_OPERATION` | 404 | No operator with that id |
| `PARAM_OUT_OF_RANGE` | 422 | Value outside the declared bounds |
| `PARAM_INVALID` | 422 | Wrong type, or an unknown parameter name |
| `OP_REQUIRES_COLOR` | 422 | Operator needs three channels |
| `SECOND_IMAGE_REQUIRED` | 422 | Operator needs a reference image |
| `CANCELLED` | 409 | Superseded by a newer preview request |
| `JOB_NOT_FOUND` | 404 | Unknown job id |

---

## Images

### `POST /api/images`
`multipart/form-data` with `file`. Returns the image summary plus a `preview`
and `thumbnail` render reference.

```json
{
  "image_id": "a1b2...", "name": "photo.jpg",
  "width": 4032, "height": 3024, "channels": 3, "megapixels": 12.19,
  "proxy_width": 1600, "proxy_height": 1200, "proxy_scale": 0.3968,
  "source_format": "jpeg", "source_bit_depth": 8,
  "had_alpha": false, "was_grayscale": false,
  "exif": { "Make": "...", "ISOSpeedRatings": 400 },
  "preview": { "render_id": "...", "url": "/api/render/...", "width": 1600,
               "height": 1200, "bytes": 184213 },
  "thumbnail": { "...": "..." }
}
```

EXIF orientation is applied at decode, so portrait phone photos are upright.

### `GET /api/images` · `GET /api/images/{id}` · `DELETE /api/images/{id}`

### `GET /api/render/{render_id}`
Raw image bytes with `Cache-Control: public, max-age=31536000, immutable`.
Render ids are content-addressed, so the browser's own cache handles A/B
comparison and undo for free.

### `GET /api/samples` · `POST /api/samples/{id}`
Seven procedurally generated samples (`scene`, `lowlight`, `hazy`, `document`,
`periodic`, `chart`, `shapes`), each exercising a different part of the
platform. No binary assets are shipped.

---

## Operations

### `GET /api/operations`
The whole catalogue. This is the single source the UI builds controls from.

```json
{
  "categories": [ { "id": "tone", "label": "Tone", "count": 23, "blurb": "..." } ],
  "operations": [ {
      "id": "tone.clahe", "name": "CLAHE", "category": "tone",
      "params": [ { "name": "clip", "kind": "float", "default": 2.0,
                    "min": 0.5, "max": 8.0, "step": 0.1,
                    "scale_aware": false, "scale_inverse": false,
                    "auto": false, "help": "...", "group": "",
                    "advanced": false, "unit": "" } ],
      "summary": "...", "formula_latex": "...", "when_to_use": "...",
      "references": ["Zuiderveld, 1994"],
      "domain": "srgb", "is_view": false, "syllabus": true,
      "requires": {}, "tags": [], "cost": "fast", "has_auto": false
  } ],
  "total": 239, "syllabus_count": 80,
  "blend_modes": [ { "value": "multiply", "label": "Multiply" } ]
}
```

**Parameter kinds** map one-to-one onto UI controls: `float`, `int`, `odd_int`,
`bool`, `enum`, `color`, `point`, `points`, `rect`, `line`, `range`, `matrix`,
`curve`, `image_ref`, `mask_ref`, `freq_mask`, `lut_file`, `seed`, `string`.

`scale_aware` means the value is a pixel length and is rescaled on
full-resolution export. `scale_inverse` additionally means it scales inversely
(normalised frequency cutoffs). `depends_on` hides a control until another
parameter takes a given value.

### `GET /api/operations/{id}`
One operator plus its `defaults` object.

### `POST /api/operations/{id}/auto`
Computes `auto`-capable parameters from the image — for example Canny's
hysteresis thresholds from the Otsu level.

```json
{ "image_id": "...", "ops": [], "target": "proxy" }
```

---

## Pipeline

### `POST /api/pipeline/preview`

```json
{
  "image_id": "a1b2...",
  "ops": [
    { "op_id": "smooth.bilateral", "params": { "sigma_color": 0.08 },
      "enabled": true, "opacity": 1.0, "blend": "normal",
      "mask": { "kind": "luminosity", "range": [0.25, 0.75], "feather": 0.05 },
      "uid": "op-1" }
  ],
  "target": "proxy",
  "seq": 42,
  "session": "client-abc",
  "include_stats": true,
  "include_overlays": true
}
```

Returns a render reference, per-operator timings, cache-hit count, any view
data, overlay renders and post-render statistics (256-bin histograms per
channel, mean, std, clipping fractions).

`seq` must increase monotonically per `session`. A newer request cancels the
one in flight, and a response whose `seq` is older than the last applied one
should be discarded by the client.

`target` is `"proxy"` (default) or `"full"`. With `"full"`, `scale_aware`
parameters are rescaled by the proxy-to-full ratio.

**Masks**: `kind` is one of `luminosity`, `color_range`, `gradient`, `rect`,
`brush`, `bitmap`. The effect is exactly zero where the mask is zero.

### `POST /api/pipeline/probe`
Statistics only, no render — for live histogram updates.

### `GET /api/pipeline/cache` · `POST /api/pipeline/cache/clear`

---

## Analysis and Advisor

### `POST /api/analysis`
`{ "image_id", "ops"?, "groups"? }` → the measurement suite plus dimension
scores. `groups` selects from `exposure`, `contrast`, `color`, `sharpness`,
`noise`, `artifacts`, `geometry`, `uniformity`, `texture`.

### `GET /api/analysis/schema`
What the Advisor measures, how each metric is scored (ideal and fail bounds,
weights), the overall weighting, and the rule table. Useful for building a
custom readout, and for citing thresholds in a write-up.

### `POST /api/advisor/diagnose`

```json
{ "image_id": "...", "strength": 1.0, "include_opt_in": false,
  "include_maps": true }
```

Returns `scores`, `findings`, `recipe`, the raw `metrics`, evidence `maps`
(clipping overlay, noise map, saliency, annotated spectrum, dark channel), and
`document_mode` / `monochrome` flags.

A finding:

```json
{
  "id": "EXP_UNDER", "dimension": "exposure", "severity": "moderate",
  "confidence": 0.86, "title": "Underexposed",
  "explanation": "Mean luminance is 0.21, below the 0.35-0.60 range ...",
  "evidence": { "metric": "mean_y", "value": 0.21, "expected": [0.35, 0.60],
                "secondary": { "clip_highlight": 0.001 } },
  "warning_only": false, "opt_in": false, "tags": [],
  "fix": {
    "label": "Raise exposure",
    "ops": [ { "op_id": "tone.exposure", "params": { "stops": 1.1 } } ],
    "derivation": "stops = log2(0.45 / 0.21) = 1.10, clamped to +/-2.5 ..."
  }
}
```

`strength` scales amount-like parameters (0.5 / 1.0 / 1.6 for Subtle /
Balanced / Strong) and leaves structural ones untouched. `include_opt_in` adds
geometric fixes, which crop and resample and so are a compositional choice.

Findings below `confidence` 0.45 should be grouped separately rather than shown
as actionable.

### `POST /api/advisor/verify`
`{ "image_id", "ops", "attribute_regressions": true }` → `overall_before`,
`overall_after`, per-dimension deltas, `regressions` (each optionally naming
the operator most likely responsible), `tradeoffs` (dimensions a given operator
family is *expected* to cost), `metrics_moved` and a prose `summary`.

---

## Augmentation

### `POST /api/augment/preview`
`{ "image_id", "stages": [{ "op_id", "params", "p" }], "count", "seed" }` →
N thumbnail variants, each with the exact recipe that produced it. The same
seed always reproduces the same set.

### `POST /api/augment/generate`
Batch job: `{ "image_ids", "stages", "variants_per_image", "seed", "format",
"quality" }` → `{ "job_id", "total" }`. The resulting zip contains the images
plus a `manifest.json` recording the realised parameters of every output, so
any image can be reproduced by replaying its recipe.

### `GET /api/augment/presets`
Seven shipped recipes: classification (light and heavy), detection
(geometry-safe), OCR/document, low-light robustness, medical-conservative,
sensor-simulation.

### `GET /api/augment/download/{id}`

---

## Export, batch and jobs

### `POST /api/export`
`{ "image_id", "ops", "format", "quality", "bit_depth", "resize_long_edge" }`
→ `{ "job_id" }`. Renders at native resolution with `scale_aware` parameters
rescaled. Formats: `png`, `jpeg`, `webp`, `tiff`, `bmp`; `bit_depth` 8 or 16
for PNG and TIFF.

### `POST /api/batch/apply`
One recipe across many images, each with its own scale handling, zipped with a
manifest.

### `GET /api/jobs` · `GET /api/jobs/{id}` · `POST /api/jobs/{id}/cancel`

```json
{ "job_id": "...", "kind": "export", "status": "running",
  "progress": 0.75, "message": "Encoding", "elapsed": 2.41, "result": null }
```

`status` is `pending`, `running`, `done`, `error` or `cancelled`.

---

## Report

### `POST /api/report`
`{ "image_id", "ops", "title", "include_advisor", "include_images" }` →
`{ "markdown", "scores_before", "scores_after", "metrics_before",
"metrics_after", "operator_count", "syllabus_operator_count" }`.

The markdown contains the input summary, the applied recipe with each
operator's theory, formula and parameters, before/after images as data URIs,
the full metric table with deltas, the Advisor's findings with their
derivations, the verification result, and the no-learned-models declaration.

### `POST /api/report/markdown`
The same, as a `text/markdown` download.

---

## WebSocket

### `WS /api/ws/session/{sid}`
Send `{"type":"watch","job_id":"..."}` to follow a job. Receives `job` frames
for watched jobs and periodic `stats` frames with cache and store occupancy.
`{"type":"ping"}` gets a `pong`.

---

## Health

### `GET /api/health`

```json
{ "status": "ok", "operations": 239, "syllabus_operations": 80,
  "categories": 13, "preview_max_edge": 1600, "max_megapixels": 40.0,
  "uses_learned_models": false }
```
