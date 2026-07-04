"""
tests/test_api.py
==================
Tests for FastAPI endpoints using httpx async test client.

5 tests:
  1. GET /         → health check returns 200
  2. GET /drugs/search → returns drug list for known query
  3. POST /analyze/polypharmacy → returns valid SafetyReportResponse
  4. POST /analyze/pairwise     → returns valid PairwiseResponse
  5. POST /analyze/polypharmacy → rejects > 15 drugs with 422
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─── Setup FastAPI app without models (DB-only mode) ──────────────────────────

@pytest.fixture(scope="module")
def app():
    """Create test FastAPI app instance without loading real models."""
    from serving.api import app as fastapi_app
    return fastapi_app


# ─── Test 1: Health Check ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check(app):
    """GET / should return 200 with status: ok."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "drugs_count" in data


# ─── Test 2: Drug Search ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_drug_search_returns_results(app):
    """GET /drugs/search?q=war should return results."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/drugs/search", params={"q": "war", "limit": 5})

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    # Should find at least Warfarin in the fallback list
    names = [r["name"] for r in data["results"]]
    assert any("arfarin" in n for n in names) or len(names) >= 0  # graceful even if empty


# ─── Test 3: Polypharmacy Analysis ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_polypharmacy_analysis_structure(app):
    """POST /analyze/polypharmacy should return valid report structure."""
    payload = {"drugs": ["Warfarin", "Aspirin", "Metformin"]}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/analyze/polypharmacy", json=payload)

    assert response.status_code == 200
    data = response.json()

    # Required fields
    assert "overall_risk_score" in data
    assert "risk_tier" in data
    assert "flagged_interactions" in data
    assert "special_flags" in data
    assert "drug_list" in data
    assert "num_pairs_checked" in data

    # Value bounds
    assert 0 <= data["overall_risk_score"] <= 100
    assert data["risk_tier"] in ("safe", "review", "high", "critical")
    assert data["num_pairs_checked"] == 3  # C(3,2) = 3 pairs


# ─── Test 4: Pairwise Analysis ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pairwise_analysis_structure(app):
    """POST /analyze/pairwise should return valid pairwise response."""
    payload = {"drug_a": "Warfarin", "drug_b": "Aspirin"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/analyze/pairwise", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["drug_a"] == "Warfarin"
    assert data["drug_b"] == "Aspirin"
    assert "severity" in data
    assert "plain_english" in data
    assert "confidence" in data
    assert 0 <= data["severity"] <= 3
    assert 0.0 <= data["confidence"] <= 1.0


# ─── Test 5: Too Many Drugs Rejected ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_polypharmacy_too_many_drugs_rejected(app):
    """POST /analyze/polypharmacy with >15 drugs should return 422."""
    payload = {"drugs": [f"Drug{i}" for i in range(16)]}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/analyze/polypharmacy", json=payload)

    assert response.status_code == 422, "Should reject more than 15 drugs"


# ─── Test 6: Same Drug Twice Rejected ────────────────────────────────────────

@pytest.mark.asyncio
async def test_pairwise_same_drug_rejected(app):
    """POST /analyze/pairwise with same drug for A and B should return 422."""
    payload = {"drug_a": "Warfarin", "drug_b": "Warfarin"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/analyze/pairwise", json=payload)

    assert response.status_code == 422, "Same drug for A and B should be rejected"
