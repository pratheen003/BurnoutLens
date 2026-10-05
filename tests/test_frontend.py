"""Tests for BurnoutLens frontend static mounting and offline asset integrity."""

from pathlib import Path
import re
import pytest
from fastapi.testclient import TestClient

from burnoutlens.api import app

client = TestClient(app)


def test_root_returns_html():
    """GET / must return HTTP 200 and the index.html page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "BurnoutLens" in response.text
    assert "<title>BurnoutLens" in response.text


def test_api_routes_preserved_after_mount():
    """Verify API endpoints continue functioning and are not shadowed by the static mount."""
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_meta = client.get("/meta")
    assert r_meta.status_code == 200
    assert "feature_schema" in r_meta.json()

    r_docs = client.get("/docs")
    assert r_docs.status_code == 200

    r_openapi = client.get("/openapi.json")
    assert r_openapi.status_code == 200

    r_clusters = client.get("/analytics/clusters")
    assert r_clusters.status_code == 200

    r_importance = client.get("/analytics/importance")
    assert r_importance.status_code == 200

    r_models = client.get("/models/comparison")
    assert r_models.status_code == 200


def test_static_assets_served():
    """Verify that CSS and JS static assets are served properly."""
    r_css = client.get("/css/styles.css")
    assert r_css.status_code == 200
    assert "text/css" in r_css.headers["content-type"]
    assert ":root" in r_css.text

    r_js = client.get("/js/app.js")
    assert r_js.status_code == 200
    assert "javascript" in r_css.headers.get("content-type", "") or "javascript" in r_js.headers["content-type"]
    assert "DATASET_EXAMPLES" in r_js.text


def test_index_html_contains_disclaimer():
    """Ensure index.html prominently displays the mandatory educational disclaimer."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Educational estimate, not a medical diagnosis." in response.text
    assert "Lifestyle-based burnout risk estimate for educational purposes. Not a medical diagnosis." in response.text


def test_no_external_urls_in_frontend():
    """Ensure no external http(s):// resource URLs are referenced in frontend HTML/CSS/JS.

    All styling and scripts must be offline-capable with no CDN requests.
    Only W3C XML namespace strings or comments are permitted.
    """
    root_dir = Path(__file__).resolve().parent.parent
    frontend_dir = root_dir / "frontend"
    assert frontend_dir.exists(), "frontend directory must exist"

    # Pattern matching http:// or https:// followed by non-whitespace
    url_pattern = re.compile(r'https?://[^\s\'"<>]+', re.IGNORECASE)

    violations = []
    for file_path in frontend_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in [".html", ".css", ".js"]:
            content = file_path.read_text(encoding="utf-8")
            matches = url_pattern.findall(content)
            for match in matches:
                # Allow standard W3C XML namespace string for SVG elements
                if match == "http://www.w3.org/2000/svg":
                    continue
                violations.append(f"{file_path.name}: {match}")

    assert not violations, f"Forbidden external URLs found in frontend assets: {violations}"
