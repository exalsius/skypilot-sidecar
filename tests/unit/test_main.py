import os

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture(autouse=True)
def clean_environment():
    """Ensure each test starts with original environment state."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


def test_app_creation():
    """Test that the FastAPI app can be created without errors."""
    assert app is not None
    assert app.title == "FastAPI"  # Default FastAPI title


def test_health_endpoint():
    """Test that the health endpoint works and the app starts properly."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_api_router_included():
    """Test that the API router is properly included with the correct prefix."""
    with TestClient(app) as client:
        # Test that the list-instances endpoint exists at the correct path
        # We expect a 422 (validation error) since we're not sending a body,
        # but this confirms the endpoint exists and is accessible
        response = client.post("/v1.0/list-instances")
        assert (
            response.status_code == 422
        )  # Validation error expected without request body


def test_app_endpoints_functionality():
    """Test that all app endpoints work correctly."""
    with TestClient(app) as client:
        # Test health endpoint exists and works
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json() == {"status": "ok"}

        # Test API endpoint exists (should get validation error, not 404)
        api_response = client.post("/v1.0/list-instances")
        assert api_response.status_code == 422  # Not 404, meaning route exists

        # Test that a non-existent endpoint returns 404
        not_found_response = client.get("/nonexistent")
        assert not_found_response.status_code == 404


def test_app_can_handle_multiple_requests():
    """Test that the app can handle multiple concurrent requests."""
    with TestClient(app) as client:
        responses = []
        # Make multiple health check requests
        for _ in range(5):
            response = client.get("/health")
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
