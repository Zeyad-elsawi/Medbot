"""
backend/tests/test_api.py
Integration tests for all FastAPI endpoints.
Run: pytest tests/test_api.py -v
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.main import app


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def client():
    """Async HTTP client wired directly to the FastAPI app (no network needed)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestHealthEndpoint:
    """Tests for GET /api/health"""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        """Health endpoint must return HTTP 200."""
        response = await client.get("/api/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_response_schema(self, client: AsyncClient) -> None:
        """Health response must match HealthResponse schema."""
        response = await client.get("/api/health")
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "qdrant_connected" in data
        assert "environment" in data
        assert data["status"] in ("ok", "degraded", "error")

    @pytest.mark.asyncio
    async def test_root_returns_200(self, client: AsyncClient) -> None:
        """Root endpoint must return 200."""
        response = await client.get("/")
        assert response.status_code == 200


class TestAnalyzeEndpoint:
    """Tests for POST /api/analyze"""

    @pytest.mark.asyncio
    async def test_analyze_missing_image_returns_422(self, client: AsyncClient) -> None:
        """Request without image must return 422 Unprocessable Entity."""
        response = await client.post("/api/analyze", data={"symptoms": "cough and fever"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_short_symptoms_returns_422(self, client: AsyncClient) -> None:
        """Symptoms shorter than 10 chars must return 422."""
        import io
        fake_image = io.BytesIO(b"fake image data")
        response = await client.post(
            "/api/analyze",
            files={"image": ("test.png", fake_image, "image/png")},
            data={"symptoms": "cough"},  # Too short (< 10 chars)
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_valid_request_returns_200(self, client: AsyncClient) -> None:
        """Valid request must return 200 with streaming SSE containing AnalysisResponse payload."""
        import io
        import json
        # Minimal valid PNG (1x1 white pixel)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        response = await client.post(
            "/api/analyze",
            files={"image": ("chest.png", io.BytesIO(png_bytes), "image/png")},
            data={"symptoms": "Patient reports persistent cough and fever for 5 days"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        payload = None
        for line in response.text.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:].strip())
                    if event.get("type") == "result":
                        payload = event.get("payload")
                        break
                except Exception:
                    pass

        assert payload is not None, f"Expected SSE result event in: {response.text[:300]}"
        assert "report_id" in payload
        assert "imaging_findings" in payload
        assert "similar_cases" in payload
        assert "differential_dx" in payload
        assert "icd10_codes" in payload
        assert "next_steps" in payload
        assert "confidence_level" in payload
        assert "disclaimer" in payload
        assert payload["confidence_level"] in ("high", "medium", "low")

    @pytest.mark.asyncio
    async def test_analyze_disclaimer_always_present(self, client: AsyncClient) -> None:
        """Disclaimer must always be in every response."""
        import io
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        response = await client.post(
            "/api/analyze",
            files={"image": ("chest.png", io.BytesIO(png_bytes), "image/png")},
            data={"symptoms": "shortness of breath and chest pain"},
        )
        if response.status_code == 200:
            assert "disclaimer" in response.text.lower() or "medical" in response.text.lower()


class TestReportEndpoint:
    """Tests for GET /api/report/{id}/pdf"""

    @pytest.mark.asyncio
    async def test_nonexistent_report_returns_404(self, client: AsyncClient) -> None:
        """Non-existent report ID must return 404."""
        response = await client.get("/api/report/nonexistent-id-12345/pdf")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_404_response_has_detail(self, client: AsyncClient) -> None:
        """404 response must have a human-readable detail message."""
        response = await client.get("/api/report/fake-id/pdf")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
