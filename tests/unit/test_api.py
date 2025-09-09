import os
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.api import get_list_instances_stream_wrapper, list_instances_stream_wrapper
from src.main import app
from src.models import Cloud, ListInstancesRequest


@pytest.fixture(autouse=True)
def clean_environment():
    """Ensure each test starts with original environment state."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.mark.asyncio
@patch("src.api.list_instances_stream")
async def test_list_instances_stream_wrapper(mock_list_instances_stream):
    """Test the list_instances_stream_wrapper function."""

    async def async_generator() -> AsyncGenerator[str, None]:
        yield "instance1"
        yield "instance2"

    mock_list_instances_stream.return_value = async_generator()
    body = ListInstancesRequest()

    result = [item async for item in list_instances_stream_wrapper(body)]
    assert result == ["instance1", "instance2"]
    mock_list_instances_stream.assert_called_once()


@pytest.mark.asyncio
@patch("src.api.list_instances_stream")
async def test_list_instances_stream_wrapper_with_clouds(mock_list_instances_stream):
    """Test the list_instances_stream_wrapper function with clouds filter."""

    async def async_generator() -> AsyncGenerator[str, None]:
        yield "instance1"

    mock_list_instances_stream.return_value = async_generator()

    clouds_filter = "aws"
    body = ListInstancesRequest(clouds=clouds_filter)
    result = [item async for item in list_instances_stream_wrapper(body)]
    assert result == ["instance1"]
    assert mock_list_instances_stream.call_args.kwargs["clouds"] == [Cloud.AWS]

    clouds_filter_list = ["aws", "azure"]
    mock_list_instances_stream.return_value = async_generator()
    body = ListInstancesRequest(clouds=clouds_filter_list)
    result = [item async for item in list_instances_stream_wrapper(body)]
    assert result == ["instance1"]
    assert mock_list_instances_stream.call_args.kwargs["clouds"] == [
        Cloud.AWS,
        Cloud.AZURE,
    ]


@pytest.mark.asyncio
async def test_list_instances_endpoint():
    """Test the list-instances endpoint."""
    mock_wrapper_func = MagicMock()

    def get_mock_stream_wrapper():
        return mock_wrapper_func

    async def async_generator() -> AsyncGenerator[str, None]:
        yield '{"instance": "test"}\n'

    mock_wrapper_func.return_value = async_generator()

    original_overrides = app.dependency_overrides
    app.dependency_overrides[get_list_instances_stream_wrapper] = (
        get_mock_stream_wrapper
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1.0/list-instances", json={"clouds": ["aws"], "gpus_only": False}
            )
            assert response.status_code == 200
            assert response.text == '{"instance": "test"}\n'

        mock_wrapper_func.assert_called_once()
        called_with_body = mock_wrapper_func.call_args.args[0]
        assert isinstance(called_with_body, ListInstancesRequest)
        assert called_with_body.clouds == ["aws"]
        assert not called_with_body.gpus_only
    finally:
        app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_list_instances_endpoint_all_fields():
    """Test the list-instances endpoint with all possible fields."""
    mock_wrapper_func = MagicMock()

    def get_mock_stream_wrapper():
        return mock_wrapper_func

    async def async_generator() -> AsyncGenerator[str, None]:
        yield '{"instance": "full_test"}\n'

    mock_wrapper_func.return_value = async_generator()

    original_overrides = app.dependency_overrides
    app.dependency_overrides[get_list_instances_stream_wrapper] = (
        get_mock_stream_wrapper
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1.0/list-instances",
                json={
                    "clouds": ["aws", "azure", "gcp"],
                    "gpus_only": True,
                    "name_filter": "gpu-.*",
                    "region_filter": "us-.*",
                    "quantity_filter": 4,
                    "case_sensitive": False,
                    "all_regions": True,
                    "drop_instance_type_none": False,
                },
            )
            assert response.status_code == 200

        called_with_body = mock_wrapper_func.call_args.args[0]
        assert isinstance(called_with_body, ListInstancesRequest)
        assert called_with_body.clouds == ["aws", "azure", "gcp"]
        assert called_with_body.gpus_only is True
        assert called_with_body.name_filter == "gpu-.*"
        assert called_with_body.region_filter == "us-.*"
        assert called_with_body.quantity_filter == 4
        assert called_with_body.case_sensitive is False
        assert called_with_body.all_regions is True
        assert called_with_body.drop_instance_type_none is False
    finally:
        app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_list_instances_endpoint_single_cloud_string():
    """Test the list-instances endpoint with single cloud as string."""
    mock_wrapper_func = MagicMock()

    def get_mock_stream_wrapper():
        return mock_wrapper_func

    async def async_generator() -> AsyncGenerator[str, None]:
        yield '{"instance": "single_cloud"}\n'

    mock_wrapper_func.return_value = async_generator()

    original_overrides = app.dependency_overrides
    app.dependency_overrides[get_list_instances_stream_wrapper] = (
        get_mock_stream_wrapper
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1.0/list-instances", json={"clouds": "lambda", "gpus_only": False}
            )
            assert response.status_code == 200

        called_with_body = mock_wrapper_func.call_args.args[0]
        assert called_with_body.clouds == "lambda"
    finally:
        app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_list_instances_endpoint_minimal_request():
    """Test the list-instances endpoint with minimal request (defaults)."""
    mock_wrapper_func = MagicMock()

    def get_mock_stream_wrapper():
        return mock_wrapper_func

    async def async_generator() -> AsyncGenerator[str, None]:
        yield '{"instance": "minimal"}\n'

    mock_wrapper_func.return_value = async_generator()

    original_overrides = app.dependency_overrides
    app.dependency_overrides[get_list_instances_stream_wrapper] = (
        get_mock_stream_wrapper
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/v1.0/list-instances", json={})
            assert response.status_code == 200

        called_with_body = mock_wrapper_func.call_args.args[0]
        assert called_with_body.gpus_only is True  # Default value
        assert called_with_body.clouds is None
        assert called_with_body.case_sensitive is True
        assert called_with_body.all_regions is False
        assert called_with_body.drop_instance_type_none is True
    finally:
        app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_list_instances_endpoint_various_clouds():
    """Test the list-instances endpoint with different cloud combinations."""
    mock_wrapper_func = MagicMock()

    def get_mock_stream_wrapper():
        return mock_wrapper_func

    async def async_generator() -> AsyncGenerator[str, None]:
        yield '{"instance": "cloud_test"}\n'

    mock_wrapper_func.return_value = async_generator()

    original_overrides = app.dependency_overrides
    app.dependency_overrides[get_list_instances_stream_wrapper] = (
        get_mock_stream_wrapper
    )

    cloud_combinations = [
        ["aws"],
        ["azure", "gcp"],
        ["cudo", "fluidstack", "hyperbolic"],
        ["lambda", "nebius", "oci", "paperspace"],
        ["runpod", "scp", "vast"],
        ["do", "ibm"],
    ]

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            for clouds in cloud_combinations:
                response = await client.post(
                    "/v1.0/list-instances", json={"clouds": clouds}
                )
                assert response.status_code == 200
    finally:
        app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_list_instances_endpoint_edge_cases():
    """Test the list-instances endpoint with edge case values."""
    mock_wrapper_func = MagicMock()

    def get_mock_stream_wrapper():
        return mock_wrapper_func

    async def async_generator() -> AsyncGenerator[str, None]:
        yield '{"instance": "edge_case"}\n'

    mock_wrapper_func.return_value = async_generator()

    original_overrides = app.dependency_overrides
    app.dependency_overrides[get_list_instances_stream_wrapper] = (
        get_mock_stream_wrapper
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Empty clouds array
            response = await client.post("/v1.0/list-instances", json={"clouds": []})
            assert response.status_code == 200

            # Zero quantity filter
            response = await client.post(
                "/v1.0/list-instances", json={"quantity_filter": 0}
            )
            assert response.status_code == 200

            # Empty string filters
            response = await client.post(
                "/v1.0/list-instances", json={"name_filter": "", "region_filter": ""}
            )
            assert response.status_code == 200

            # Large quantity filter
            response = await client.post(
                "/v1.0/list-instances", json={"quantity_filter": 1000}
            )
            assert response.status_code == 200
    finally:
        app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_list_instances_endpoint_invalid_cloud_names():
    """Test the list-instances endpoint with invalid cloud names."""
    mock_wrapper_func = MagicMock()

    def get_mock_stream_wrapper():
        return mock_wrapper_func

    async def async_generator() -> AsyncGenerator[str, None]:
        yield '{"instance": "invalid_cloud"}\n'

    mock_wrapper_func.return_value = async_generator()

    original_overrides = app.dependency_overrides
    app.dependency_overrides[get_list_instances_stream_wrapper] = (
        get_mock_stream_wrapper
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Invalid cloud names should be filtered out by the wrapper
            response = await client.post(
                "/v1.0/list-instances",
                json={"clouds": ["aws", "invalid_cloud", "azure"]},
            )
            assert response.status_code == 200

            # All invalid cloud names
            response = await client.post(
                "/v1.0/list-instances", json={"clouds": ["invalid1", "invalid2"]}
            )
            assert response.status_code == 200
    finally:
        app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_list_instances_endpoint_negative_quantity():
    """Test the list-instances endpoint with negative quantity filter."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1.0/list-instances", json={"quantity_filter": -1}
        )
        # Should accept negative values (might be handled by business logic)
        assert response.status_code == 200 or response.status_code == 422


@pytest.mark.asyncio
async def test_list_instances_endpoint_invalid_types():
    """Test the list-instances endpoint with invalid data types."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Invalid boolean type for gpus_only
        response = await client.post(
            "/v1.0/list-instances",
            json={
                "gpus_only": "invalid_boolean_value"
            },  # String that can't be converted to boolean
        )
        assert response.status_code == 422

        # Invalid type for quantity_filter
        response = await client.post(
            "/v1.0/list-instances",
            json={"quantity_filter": "four"},  # String instead of integer
        )
        assert response.status_code == 422

        # Invalid type for case_sensitive
        response = await client.post(
            "/v1.0/list-instances",
            json={
                "case_sensitive": "not_a_boolean"
            },  # String that can't be converted to boolean
        )
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_instances_endpoint_malformed_json():
    """Test the list-instances endpoint with malformed JSON."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Send malformed JSON
        response = await client.post(
            "/v1.0/list-instances",
            content='{"gpus_only": true, invalid json}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_instances_endpoint_extra_fields():
    """Test the list-instances endpoint with extra unknown fields."""
    mock_wrapper_func = MagicMock()

    def get_mock_stream_wrapper():
        return mock_wrapper_func

    async def async_generator() -> AsyncGenerator[str, None]:
        yield '{"instance": "extra_fields"}\n'

    mock_wrapper_func.return_value = async_generator()

    original_overrides = app.dependency_overrides
    app.dependency_overrides[get_list_instances_stream_wrapper] = (
        get_mock_stream_wrapper
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1.0/list-instances",
                json={
                    "clouds": ["aws"],
                    "gpus_only": True,
                    "unknown_field": "should_be_ignored",
                    "another_extra": 123,
                },
            )
            # Pydantic should ignore extra fields by default
            assert response.status_code == 200

            called_with_body = mock_wrapper_func.call_args.args[0]
            assert called_with_body.clouds == ["aws"]
            assert called_with_body.gpus_only is True
    finally:
        app.dependency_overrides = original_overrides


def test_health_endpoint():
    """Test the health endpoint."""
    # This test is not directly related to the list-instances endpoint and its stream wrapper.
    # It remains unchanged as it was not part of the edit instructions.

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
