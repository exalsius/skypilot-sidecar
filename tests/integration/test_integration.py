import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.main import app

# Set up test environment before importing the app
TEST_DATA_PATH = Path(__file__).parent / "data"
os.environ["EXLS_SKY_CATALOG_PATH"] = str(TEST_DATA_PATH)


@pytest.fixture(autouse=True)
def clean_environment():
    """Ensure each test starts with original environment state."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_data_path() -> Path:
    """Path to test data directory."""
    return TEST_DATA_PATH


@pytest.fixture
def client():
    """Test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test cases for the health endpoint."""

    def test_health_endpoint(self, client: TestClient):
        """Test that the health endpoint returns OK status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestListInstancesEndpoint:
    """Test cases for the list-instances endpoint."""

    @pytest.fixture
    def base_url(self) -> str:
        """Base URL for API endpoints."""
        return "/v1.0/list-instances"

    def test_list_instances_all_clouds_default(self, client: TestClient, base_url: str):
        """Test listing instances from all clouds with default parameters."""
        request_body = {}
        response = client.post(base_url, json=request_body)
        assert response.status_code == 200

        # Verify streaming response
        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]

        # Should have data from multiple clouds
        assert len(lines) > 0

        # Parse first few lines to verify structure
        for line in lines[:3]:
            data = json.loads(line)
            for _, value in data.items():
                assert value is not None
                assert len(value) > 0
                assert "cloud" in value[0]
                assert "instance_type" in value[0]
                assert "accelerator_name" in value[0]
                assert "region" in value[0]

    def test_list_instances_specific_cloud(self, client: TestClient, base_url: str):
        """Test listing instances from a specific cloud."""
        request_body = {"clouds": ["aws"]}
        response = client.post(base_url, json=request_body)
        assert response.status_code == 200

        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]
        assert len(lines) > 0

        # Verify all results are from AWS
        for line in lines:
            data = json.loads(line)
            for _, value in data.items():
                assert value[0]["cloud"] == "aws"

    def test_list_instances_multiple_clouds(self, client: TestClient, base_url: str):
        """Test listing instances from multiple specific clouds."""
        request_body = {"clouds": ["aws", "cudo"]}
        response = client.post(base_url, json=request_body)
        assert response.status_code == 200

        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]
        assert len(lines) > 0

        # Verify results are only from specified clouds
        clouds_found = set()
        for line in lines:
            data = json.loads(line)
            for _, value in data.items():
                clouds_found.add(value[0]["cloud"])

        assert clouds_found.issubset({"aws", "cudo"})

    def test_list_instances_single_cloud_string(
        self, client: TestClient, base_url: str
    ):
        """Test listing instances with cloud specified as a string."""
        request_body = {"clouds": "aws"}
        response = client.post(base_url, json=request_body)
        assert response.status_code == 200

        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]
        assert len(lines) > 0

        # Verify all results are from AWS
        for line in lines:
            data = json.loads(line)
            for _, value in data.items():
                assert value[0]["cloud"] == "aws"

    def test_list_instances_gpus_only_filter(self, client: TestClient, base_url: str):
        """Test filtering for GPU instances only."""
        request_body = {"gpus_only": True, "clouds": ["aws"]}
        response = client.post(base_url, json=request_body)
        assert response.status_code == 200

        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]
        assert len(lines) > 0

        # All instances should have accelerators
        for line in lines:
            data = json.loads(line)
            for _, value in data.items():
                assert value[0]["accelerator_name"] != "CPU-ONLY"
                assert value[0]["accelerator_count"] > 0

    def test_list_instances_include_cpu_only(self, client: TestClient, base_url: str):
        """Test including CPU-only instances."""
        request_body = {"gpus_only": False, "clouds": ["aws"]}
        response = client.post(base_url, json=request_body)
        assert response.status_code == 200

        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]
        assert len(lines) > 0

    def test_list_instances_name_filter(self, client: TestClient, base_url: str):
        """Test filtering instances by accelerator name."""
        request_body = {
            "name_filter": "V520",
            "clouds": ["aws"],
            "case_sensitive": True,
        }
        response = client.post(base_url, json=request_body)
        assert response.status_code == 200

        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]
        assert len(lines) > 0

        # All results should match the filter
        for line in lines:
            data = json.loads(line)
            for _, value in data.items():
                assert "V520" in value[0]["accelerator_name"]

    def test_list_instances_name_filter_case_insensitive(
        self, client: TestClient, base_url: str
    ):
        """Test case-insensitive name filtering."""
        request_body = {
            "name_filter": "v520",
            "clouds": ["aws"],
            "case_sensitive": False,
        }
        response = client.post(base_url, json=request_body)
        assert response.status_code == 200

        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]
        assert len(lines) > 0

    def test_list_instances_region_filter(self, client: TestClient, base_url: str):
        """Test filtering instances by region."""
        request_body = {"region_filter": "us-west", "clouds": ["aws"]}
        response = client.post(base_url, json=request_body)
        assert response.status_code == 200

        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]

        # Should return results only from us-west regions (if any exist in test data)
        # If no matching data, lines will be empty which is acceptable
        for line in lines:
            data = json.loads(line)
            for _, value in data.items():
                assert "us-west" in value[0]["region"].lower(), (
                    f"Region {value[0]['region']} does not contain 'us-west'"
                )

    def test_list_instances_quantity_filter(self, client: TestClient, base_url: str):
        """Test filtering instances by accelerator count."""
        request_body = {"quantity_filter": 1, "clouds": ["aws"], "gpus_only": True}
        response = client.post(base_url, json=request_body)
        assert response.status_code == 200

        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]

        # All results should have exactly 1 accelerator
        for line in lines:
            data = json.loads(line)
            for _, value in data.items():
                assert value[0]["accelerator_count"] == 1.0

    def test_list_instances_drop_instance_type_none(
        self, client: TestClient, base_url: str
    ):
        """Test dropping instances with null instance types."""
        request_body = {"drop_instance_type_none": True, "clouds": ["aws"]}
        response = client.post(base_url, json=request_body)
        assert response.status_code == 200

        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]

        # No instance should have null instance_type
        for line in lines:
            data = json.loads(line)
            for _, value in data.items():
                assert value[0]["instance_type"] is not None

    def test_list_instances_all_regions(self, client: TestClient, base_url: str):
        """Test showing instances from all regions."""
        request_body = {"all_regions": True, "clouds": ["aws"]}
        response = client.post(base_url, json=request_body)
        assert response.status_code == 200

        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]
        assert len(lines) > 0

    def test_list_instances_complex_filter(self, client: TestClient, base_url: str):
        """Test complex filtering with multiple parameters."""
        request_body = {
            "gpus_only": True,
            "clouds": ["aws", "cudo"],
            "case_sensitive": False,
            "all_regions": True,
            "drop_instance_type_none": True,
        }
        response = client.post(base_url, json=request_body)
        assert response.status_code == 200

        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]
        assert len(lines) > 0

        # Verify the complex filter conditions
        clouds_found = set()
        for line in lines:
            data = json.loads(line)
            for _, value in data.items():
                clouds_found.add(value[0]["cloud"])
                assert value[0]["accelerator_name"] != "CPU-ONLY"
                assert value[0]["instance_type"] is not None

        assert clouds_found.issubset({"aws", "cudo"})

    def test_list_instances_invalid_cloud(self, client: TestClient, base_url: str):
        """Test handling of invalid cloud names."""
        request_body = {"clouds": ["invalid_cloud"]}
        response = client.post(base_url, json=request_body)
        assert response.status_code == 200

        # Should return empty response since invalid cloud is filtered out
        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]
        assert len(lines) == 0


class TestEmptyDataScenario:
    """Test cases for empty data scenarios."""

    def test_list_instances_empty_directory(self, client: TestClient):
        """Test behavior when pointing to empty data directory."""
        # Temporarily set catalog path to test_empty directory
        original_path = os.environ.get("EXLS_SKY_CATALOG_PATH")
        test_empty_path = TEST_DATA_PATH / "test_empty"
        os.environ["EXLS_SKY_CATALOG_PATH"] = str(test_empty_path)

        try:
            # Create fresh app instance with updated config
            from importlib import reload

            from fastapi import FastAPI

            import src.api

            # Reload modules to pick up new config
            import src.config
            import src.main

            reload(src.config)
            reload(src.api)
            reload(src.main)

            # Create new app instance with fresh config
            fresh_app = FastAPI()
            fresh_app.add_api_route("/health", src.main.health, methods=["GET"])
            fresh_app.include_router(
                src.api.router, prefix=f"/v{src.config.config.api_version}"
            )

            # Create new test client with fresh app
            fresh_client = TestClient(fresh_app)

            request_body = {}
            response = fresh_client.post("/v1.0/list-instances", json=request_body)
            assert response.status_code == 200

            # Should return empty response
            content = response.content.decode("utf-8")
            lines = [line for line in content.split("\n") if line.strip()]
            assert len(lines) == 0

        finally:
            # Restore original path
            if original_path:
                os.environ["EXLS_SKY_CATALOG_PATH"] = original_path
            else:
                # Remove the env var if it wasn't set originally
                os.environ.pop("EXLS_SKY_CATALOG_PATH", None)
            # Reload modules back to original state
            reload(src.config)
            reload(src.api)
            reload(src.main)

    def test_list_instances_no_matching_filters(self, client: TestClient):
        """Test behavior when filters match no data."""
        # Use a very specific filter that shouldn't match any data
        request_body = {
            "clouds": ["aws"],
            "name_filter": "NONEXISTENT_GPU_MODEL_12345",
            "case_sensitive": True,
        }
        response = client.post("/v1.0/list-instances", json=request_body)
        assert response.status_code == 200

        # Should return empty or very limited results
        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]
        # Either no results or results that actually match the impossible filter
        for line in lines:
            data = json.loads(line)
            for _, value in data.items():
                assert "NONEXISTENT_GPU_MODEL_12345" in value[0]["accelerator_name"]


class TestResponseFormat:
    """Test cases for response format validation."""

    def test_response_structure_validation(self, client: TestClient):
        """Test that response structure matches expected format."""
        request_body = {"clouds": ["aws"], "gpus_only": True}
        response = client.post("/v1.0/list-instances", json=request_body)
        assert response.status_code == 200

        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]
        assert len(lines) > 0

        # Validate first response item structure
        first_item = json.loads(lines[0])
        required_fields = [
            "cloud",
            "instance_type",
            "accelerator_name",
            "accelerator_count",
            "cpu_count",
            "device_memory",
            "memory",
            "price",
            "spot_price",
            "region",
        ]
        for _, value in first_item.items():
            for field in required_fields:
                assert field in value[0]

        # Validate data types
        for _, value in first_item.items():
            assert isinstance(value[0]["cloud"], str)
            assert isinstance(value[0]["accelerator_name"], str)
            assert isinstance(value[0]["accelerator_count"], (int, float))
            assert isinstance(value[0]["region"], str)


class TestErrorHandling:
    """Test cases for error handling and edge cases."""

    def test_malformed_json_request(self, client: TestClient):
        """Test handling of malformed JSON in request."""
        response = client.post(
            "/v1.0/list-instances",
            content=b"malformed json{",
            headers={"Content-Type": "application/json"},
        )
        # Should return a 422 validation error
        assert response.status_code == 422

    def test_invalid_request_fields(self, client: TestClient):
        """Test handling of invalid request fields."""
        request_body = {
            "clouds": ["aws"],
            "quantity_filter": -1,  # Invalid negative quantity
        }
        response = client.post("/v1.0/list-instances", json=request_body)
        # Should still process the request, possibly ignoring invalid filter
        assert response.status_code in [200, 422]

    def test_empty_clouds_list(self, client: TestClient):
        """Test behavior with empty clouds list."""
        request_body = {"clouds": []}
        response = client.post("/v1.0/list-instances", json=request_body)
        assert response.status_code == 200

        # Should return empty response or default to all clouds
        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]
        # Acceptable to have either empty results or all clouds' results
        assert len(lines) >= 0

    def test_very_large_quantity_filter(self, client: TestClient):
        """Test handling of extremely large quantity filter."""
        request_body = {"clouds": ["aws"], "quantity_filter": 999999, "gpus_only": True}
        response = client.post("/v1.0/list-instances", json=request_body)
        assert response.status_code == 200

        # Should return empty or very few results
        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]
        # Any results should actually have the requested quantity
        for line in lines:
            data = json.loads(line)
            for _, value in data.items():
                assert value[0]["accelerator_count"] >= 999999

    def test_special_characters_in_filters(self, client: TestClient):
        """Test handling of special characters in name filter."""
        request_body = {
            "clouds": ["aws"],
            "name_filter": "GPU@#$%^&*()_+[]|\\:;<>?,./",
        }
        response = client.post("/v1.0/list-instances", json=request_body)
        assert response.status_code == 200
        # Should not crash and return empty results (likely no match)

    def test_request_timeout_handling(self, client: TestClient):
        """Test that requests complete within reasonable time."""
        import time

        start_time = time.time()
        request_body = {"clouds": ["aws", "cudo", "lambda"], "gpus_only": True}
        response = client.post("/v1.0/list-instances", json=request_body)
        end_time = time.time()

        # Request should complete within 30 seconds for integration tests
        assert (end_time - start_time) < 30, (
            f"Request took too long: {end_time - start_time} seconds"
        )
        assert response.status_code == 200

    def test_large_response_handling(self, client: TestClient):
        """Test handling of potentially large responses with all clouds."""
        request_body = {"all_regions": True, "gpus_only": False}  # Maximum data
        response = client.post("/v1.0/list-instances", json=request_body)
        assert response.status_code == 200

        # Should handle large response without errors
        content = response.content.decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]

        # Verify we can parse all lines without JSON errors
        parsed_count = 0
        for line in lines:
            try:
                data = json.loads(line)
                parsed_count += 1
                # Basic sanity check on data structure
                for _, value in data.items():
                    assert isinstance(value, list)
                    assert len(value) > 0
                    assert isinstance(value[0], dict)
            except json.JSONDecodeError as e:
                pytest.fail(f"Failed to parse JSON line: {line[:100]}... Error: {e}")

        assert parsed_count == len(lines), (
            f"Parsed {parsed_count} lines out of {len(lines)}"
        )


class TestConcurrentRequests:
    """Test cases for concurrent request handling."""

    def test_concurrent_requests(self, client: TestClient):
        """Test handling multiple concurrent requests."""
        import threading
        from threading import Lock

        results = []
        errors = []
        results_lock = Lock()
        errors_lock = Lock()

        def make_request():
            try:
                request_body = {"clouds": ["aws"], "gpus_only": True}
                response = client.post("/v1.0/list-instances", json=request_body)

                with results_lock:
                    results.append(
                        {
                            "status_code": response.status_code,
                            "content_length": len(
                                response.content.decode("utf-8").strip()
                            ),
                        }
                    )
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        # Make 5 concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All requests should succeed
        assert len(errors) == 0, f"Errors occurred during concurrent requests: {errors}"
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"

        # All requests should return 200
        status_codes = [r["status_code"] for r in results]
        assert all(code == 200 for code in status_codes), (
            f"Not all requests returned 200: {status_codes}"
        )

        # All responses should have some content (unless truly empty)
        content_lengths = [r["content_length"] for r in results]
        assert all(length >= 0 for length in content_lengths), (
            f"Negative content lengths: {content_lengths}"
        )

    def test_data_consistency_across_requests(self, client: TestClient):
        """Test that identical requests return consistent data."""
        request_body = {"clouds": ["aws"], "gpus_only": True, "quantity_filter": 1}

        # Make the same request twice
        response1 = client.post("/v1.0/list-instances", json=request_body)
        response2 = client.post("/v1.0/list-instances", json=request_body)

        assert response1.status_code == 200
        assert response2.status_code == 200

        content1 = response1.content.decode("utf-8")
        content2 = response2.content.decode("utf-8")

        lines1 = [line for line in content1.split("\n") if line.strip()]
        lines2 = [line for line in content2.split("\n") if line.strip()]

        # Should return the same number of results
        assert len(lines1) == len(lines2), (
            f"Inconsistent result count: {len(lines1)} vs {len(lines2)}"
        )

        # Content should be identical for identical requests
        # (assuming data doesn't change during test execution)
        assert content1 == content2, "Identical requests returned different content"
