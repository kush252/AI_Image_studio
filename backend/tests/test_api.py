"""API contract. PLAN.md Section 4 and Phase 10."""
from __future__ import annotations

import io
import time

import numpy as np
import pytest

from app.core import imageio as iio


@pytest.fixture(scope="module")
def uploaded(client, clean):
    """Upload one image and return its payload."""
    payload = iio.encode(clean, "png")
    r = client.post("/api/images", files={"file": ("plate.png", payload, "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------- health

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["operations"] > 200
    assert d["syllabus_operations"] > 50
    assert d["uses_learned_models"] is False


def test_catalogue_shape(client):
    r = client.get("/api/operations")
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == len(d["operations"])
    assert d["blend_modes"], "the UI needs the blend-mode list"
    for op in d["operations"]:
        assert {"id", "name", "category", "params", "summary"} <= set(op)
        for p in op["params"]:
            assert {"name", "kind", "default", "scale_aware"} <= set(p)


def test_single_operation_includes_defaults(client):
    r = client.get("/api/operations/tone.clahe")
    assert r.status_code == 200
    d = r.json()
    assert d["id"] == "tone.clahe"
    assert "clip" in d["defaults"]


def test_unknown_operation_404(client):
    r = client.get("/api/operations/nope.missing")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "UNKNOWN_OPERATION"


# ------------------------------------------------------------------ upload

def test_upload_returns_full_summary(uploaded):
    for key in ("image_id", "width", "height", "proxy_scale", "preview",
                "thumbnail", "megapixels"):
        assert key in uploaded, f"missing {key}"
    assert uploaded["preview"]["url"].startswith("/api/render/")


def test_render_is_immutably_cacheable(client, uploaded):
    r = client.get(uploaded["preview"]["url"])
    assert r.status_code == 200
    assert r.headers["cache-control"].startswith("public")
    assert "immutable" in r.headers["cache-control"]
    assert r.headers["content-type"].startswith("image/")
    assert len(r.content) > 500


def test_missing_render_404(client):
    r = client.get("/api/render/" + "0" * 32)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "RENDER_NOT_FOUND"


def test_upload_rejects_non_image(client):
    r = client.post("/api/images",
                    files={"file": ("x.png", b"not an image at all", "image/png")})
    assert r.status_code in (400, 415)
    assert r.json()["error"]["code"] in ("UNSUPPORTED_FORMAT", "DECODE_FAILED")


def test_upload_rejects_oversized(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_megapixels", 0.01)
    big = np.zeros((512, 512, 3), np.float32)
    r = client.post("/api/images",
                    files={"file": ("big.png", iio.encode(big, "png"), "image/png")})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_samples_load(client):
    r = client.get("/api/samples")
    assert r.status_code == 200
    samples = r.json()["samples"]
    assert len(samples) >= 5
    for s in samples:
        assert {"id", "name", "blurb"} <= set(s)

    loaded = client.post(f"/api/samples/{samples[0]['id']}")
    assert loaded.status_code == 200
    assert loaded.json()["width"] > 100


def test_unknown_sample_404(client):
    r = client.post("/api/samples/not-a-sample")
    assert r.status_code == 404


# ----------------------------------------------------------------- pipeline

def test_preview_roundtrip(client, uploaded):
    body = {
        "image_id": uploaded["image_id"],
        "ops": [
            {"op_id": "tone.gamma", "params": {"gamma": 0.8}},
            {"op_id": "tone.clahe", "params": {"clip": 2.0}},
        ],
        "seq": 1,
    }
    r = client.post("/api/pipeline/preview", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["seq"] == 1
    assert len(d["timings"]) == 2
    assert d["stats"]["histogram"], "the UI needs a histogram with every preview"
    assert len(d["stats"]["histogram"]) == 256
    assert d["render"]["width"] == uploaded["proxy_width"]
    assert client.get(d["render"]["url"]).status_code == 200


def test_preview_cache_hit_on_repeat(client, uploaded):
    body = {
        "image_id": uploaded["image_id"],
        "ops": [{"op_id": "smooth.gaussian", "params": {"sigma": 2.5}}],
        "seq": 2,
        "session": "cache-probe",
    }
    client.post("/api/pipeline/preview", json=body)
    second = client.post("/api/pipeline/preview", json={**body, "seq": 3})
    assert second.json()["cache_hits"] >= 1


def test_preview_with_no_ops_returns_the_original(client, uploaded):
    r = client.post("/api/pipeline/preview",
                    json={"image_id": uploaded["image_id"], "ops": [], "seq": 4})
    assert r.status_code == 200
    assert r.json()["timings"] == []


def test_preview_bad_param_is_422(client, uploaded):
    r = client.post("/api/pipeline/preview", json={
        "image_id": uploaded["image_id"],
        "ops": [{"op_id": "tone.gamma", "params": {"gamma": 999}}],
    })
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "PARAM_OUT_OF_RANGE"
    assert body["field"] == "gamma"


def test_preview_unknown_param_is_422(client, uploaded):
    r = client.post("/api/pipeline/preview", json={
        "image_id": uploaded["image_id"],
        "ops": [{"op_id": "tone.gamma", "params": {"nonsense": 1}}],
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "PARAM_INVALID"


def test_preview_unknown_image_404(client):
    r = client.post("/api/pipeline/preview",
                    json={"image_id": "deadbeef", "ops": []})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "IMAGE_NOT_FOUND"


def test_probe_returns_stats_without_a_render(client, uploaded):
    r = client.post("/api/pipeline/probe", json={
        "image_id": uploaded["image_id"],
        "ops": [{"op_id": "tone.negative", "params": {}}],
    })
    assert r.status_code == 200
    assert "render" not in r.json()
    assert r.json()["stats"]["mean"] > 0


def test_cache_stats(client):
    r = client.get("/api/pipeline/cache")
    assert r.status_code == 200
    d = r.json()
    assert {"pipeline", "renders", "store"} <= set(d)


def test_auto_params(client, uploaded):
    r = client.post("/api/operations/edge.canny/auto",
                    json={"image_id": uploaded["image_id"], "ops": []})
    assert r.status_code == 200
    params = r.json()["params"]
    assert "t1" in params and "t2" in params
    assert 0 <= params["t1"] <= params["t2"] <= 1


def test_auto_on_operator_without_auto_is_an_error(client, uploaded):
    r = client.post("/api/operations/tone.negative/auto",
                    json={"image_id": uploaded["image_id"], "ops": []})
    assert r.status_code >= 400


# ----------------------------------------------------------------- analysis

def test_analysis_returns_metrics_and_scores(client, uploaded):
    r = client.post("/api/analysis", json={"image_id": uploaded["image_id"]})
    assert r.status_code == 200
    d = r.json()
    assert "exposure" in d["metrics"]
    assert "sharpness" in d["metrics"]
    assert 0 <= d["scores"]["overall"] <= 100


def test_analysis_group_selection(client, uploaded):
    r = client.post("/api/analysis", json={
        "image_id": uploaded["image_id"], "groups": ["exposure", "noise"],
    })
    assert set(r.json()["metrics"]) == {"exposure", "noise"}


def test_analysis_schema_documents_the_rules(client):
    r = client.get("/api/analysis/schema")
    assert r.status_code == 200
    d = r.json()
    assert d["groups"]
    assert d["scoring"]["exposure"]
    assert any(rule["id"] == "EXP_UNDER" for rule in d["rules"])
    for rows in d["scoring"].values():
        for row in rows:
            assert len(row["ideal"]) == 2


# ------------------------------------------------------------------ advisor

def test_diagnose_returns_findings_and_recipe(client, uploaded):
    r = client.post("/api/advisor/diagnose", json={"image_id": uploaded["image_id"]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert "scores" in d and "findings" in d and "recipe" in d
    assert d["maps"], "the finding cards need evidence maps"
    for name, ref in d["maps"].items():
        assert client.get(ref["url"]).status_code == 200, f"map {name} unfetchable"
    for f in d["findings"]:
        assert {"id", "severity", "confidence", "explanation", "evidence"} <= set(f)
        if f["fix"]:
            assert f["fix"]["derivation"]


def test_diagnose_on_a_degraded_image_proposes_a_fix(client, clean):
    dark = np.clip(clean ** 2.4, 0, 1).astype(np.float32)
    up = client.post("/api/images",
                     files={"file": ("dark.png", iio.encode(dark, "png"), "image/png")})
    iid = up.json()["image_id"]
    d = client.post("/api/advisor/diagnose", json={"image_id": iid}).json()
    assert any(f["id"] == "EXP_UNDER" for f in d["findings"])
    assert d["recipe"], "no recipe for a clearly underexposed image"


def test_verify_endpoint(client, clean):
    dark = np.clip(clean ** 2.4, 0, 1).astype(np.float32)
    up = client.post("/api/images",
                     files={"file": ("d2.png", iio.encode(dark, "png"), "image/png")})
    iid = up.json()["image_id"]
    recipe = client.post("/api/advisor/diagnose", json={"image_id": iid}).json()["recipe"]
    r = client.post("/api/advisor/verify", json={"image_id": iid, "ops": recipe})
    assert r.status_code == 200
    v = r.json()
    assert "overall_before" in v and "overall_after" in v
    assert v["dimensions"] and v["summary"]
    assert client.get(v["render"]["url"]).status_code == 200


def test_strength_multiplier_changes_the_recipe(client, clean):
    noisy = np.clip(
        clean + np.random.default_rng(1).normal(0, 20 / 255, clean.shape), 0, 1
    ).astype(np.float32)
    up = client.post("/api/images",
                     files={"file": ("n.png", iio.encode(noisy, "png"), "image/png")})
    iid = up.json()["image_id"]
    subtle = client.post("/api/advisor/diagnose",
                         json={"image_id": iid, "strength": 0.5}).json()["recipe"]
    strong = client.post("/api/advisor/diagnose",
                         json={"image_id": iid, "strength": 1.6}).json()["recipe"]
    assert subtle and strong
    assert subtle != strong, "the strength multiplier had no effect"


# ------------------------------------------------------------------ augment

def test_augment_preview_is_reproducible(client, uploaded):
    body = {
        "image_id": uploaded["image_id"],
        "stages": [
            {"op_id": "aug.gaussian_noise", "params": {}, "p": 1.0},
            {"op_id": "aug.rotate", "params": {}, "p": 1.0},
        ],
        "count": 4,
        "seed": 7,
    }
    a = client.post("/api/augment/preview", json=body).json()
    b = client.post("/api/augment/preview", json=body).json()
    assert len(a["variants"]) == 4
    # Same seed -> same sampled parameters for every variant.
    for va, vb in zip(a["variants"], b["variants"]):
        assert va["recipe"] == vb["recipe"]
    c = client.post("/api/augment/preview", json={**body, "seed": 8}).json()
    assert c["variants"][0]["recipe"] != a["variants"][0]["recipe"]


def test_augment_presets(client):
    r = client.get("/api/augment/presets")
    assert r.status_code == 200
    presets = r.json()["presets"]
    assert len(presets) >= 5
    for p in presets:
        assert p["stages"], f"{p['id']} has no stages"


def test_augment_generate_produces_a_zip_with_manifest(client, uploaded):
    import json
    import zipfile

    r = client.post("/api/augment/generate", json={
        "image_ids": [uploaded["image_id"]],
        "stages": [{"op_id": "aug.flip", "params": {}, "p": 1.0}],
        "variants_per_image": 3,
        "seed": 5,
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    job = _await_job(client, job_id)
    assert job["status"] == "done", job
    result = job["result"]
    assert result["count"] == 3

    blob = client.get(result["url"])
    assert blob.status_code == 200
    with zipfile.ZipFile(io.BytesIO(blob.content)) as zf:
        names = zf.namelist()
        assert "manifest.json" in names, "a dataset without a manifest is not reproducible"
        assert len([n for n in names if n != "manifest.json"]) == 3
        manifest = json.loads(zf.read("manifest.json"))
        assert len(manifest["outputs"]) == 3
        for out in manifest["outputs"]:
            # The exact realised parameters must be recorded, or the output
            # cannot be reproduced.
            assert out["recipe"] and "sub_seed" in out


# ------------------------------------------------------------------- export

def _await_job(client, job_id, timeout=90.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.15)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


@pytest.mark.parametrize("fmt,depth", [("png", 8), ("jpeg", 8), ("png", 16)])
def test_export_full_resolution(client, uploaded, fmt, depth):
    r = client.post("/api/export", json={
        "image_id": uploaded["image_id"],
        "ops": [{"op_id": "tone.clahe", "params": {"clip": 2.0}}],
        "format": fmt,
        "bit_depth": depth,
    })
    assert r.status_code == 200
    job = _await_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    result = job["result"]
    assert result["width"] == uploaded["width"]
    assert result["height"] == uploaded["height"]
    blob = client.get(result["url"])
    assert blob.status_code == 200
    assert len(blob.content) > 1000


def test_batch_apply(client, uploaded, clean):
    second = client.post(
        "/api/images",
        files={"file": ("b.png", iio.encode(clean * 0.8, "png"), "image/png")},
    ).json()
    r = client.post("/api/batch/apply", json={
        "image_ids": [uploaded["image_id"], second["image_id"]],
        "ops": [{"op_id": "tone.gamma", "params": {"gamma": 0.9}}],
        "format": "jpeg",
    })
    assert r.status_code == 200
    job = _await_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["count"] == 2


def test_job_not_found(client):
    assert client.get("/api/jobs/nope").status_code == 404


# ------------------------------------------------------------------- report

def test_report_generation(client, uploaded):
    r = client.post("/api/report", json={
        "image_id": uploaded["image_id"],
        "ops": [
            {"op_id": "tone.clahe", "params": {"clip": 2.0}},
            {"op_id": "sharpen.unsharp", "params": {"amount": 0.6}},
        ],
        "include_advisor": True,
        "include_images": False,
    })
    assert r.status_code == 200
    d = r.json()
    md = d["markdown"]
    assert d["operator_count"] == 2
    # The report must carry the theory, the numbers and the ML declaration.
    for needle in ("## 1. Input", "## 2. Processing pipeline",
                   "## 3. Measured effect", "no learned models",
                   "CLAHE", "Unsharp"):
        assert needle in md, f"report is missing {needle!r}"
    assert "$$" in md, "operator formulas are missing"


def test_report_markdown_download(client, uploaded):
    r = client.post("/api/report/markdown", json={
        "image_id": uploaded["image_id"], "ops": [], "include_images": False,
    })
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert "attachment" in r.headers.get("content-disposition", "")


# ---------------------------------------------------------------- websocket

def test_websocket_streams_stats(client):
    with client.websocket_connect("/api/ws/session/test") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        # The next message should be a stats frame.
        for _ in range(4):
            msg = ws.receive_json()
            if msg["type"] == "stats":
                assert "pipeline_cache" in msg
                return
        raise AssertionError("no stats frame received")


# ---------------------------------------------------------------- lifecycle

def test_image_delete(client, clean):
    up = client.post(
        "/api/images",
        files={"file": ("tmp.png", iio.encode(clean, "png"), "image/png")},
    ).json()
    assert client.delete(f"/api/images/{up['image_id']}").status_code == 200
    r = client.post("/api/pipeline/preview",
                    json={"image_id": up["image_id"], "ops": []})
    assert r.status_code == 404


def test_list_images(client, uploaded):
    r = client.get("/api/images")
    assert r.status_code == 200
    ids = [i["image_id"] for i in r.json()["images"]]
    assert uploaded["image_id"] in ids


def test_spa_is_served_if_built(client):
    """The single-container mode serves the built frontend at the root."""
    from app.config import settings

    if not (settings.static_dir / "index.html").exists():
        pytest.skip("frontend not built")
    r = client.get("/")
    assert r.status_code == 200
    assert "<div id=\"root\">" in r.text
