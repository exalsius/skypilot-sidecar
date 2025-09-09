import os

import pytest
from pydantic import ValidationError

from src.models import Cloud, CloudFilter, InstanceTypeInfo, ListInstancesRequest


@pytest.fixture(autouse=True)
def clean_environment():
    """Ensure each test starts with original environment state."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


class TestCloudEnum:
    """Test cases for the Cloud enum."""

    def test_cloud_enum_values(self):
        """Test that all expected cloud values are present."""
        expected_clouds = [
            "aws",
            "azure",
            "cudo",
            "do",
            "fluidstack",
            "gcp",
            "hyperbolic",
            "ibm",
            "lambda",
            "nebius",
            "oci",
            "paperspace",
            "runpod",
            "scp",
            "vast",
        ]

        actual_clouds = [cloud.value for cloud in Cloud]
        assert sorted(actual_clouds) == sorted(expected_clouds)

    def test_cloud_enum_access(self):
        """Test accessing enum values by name and value."""
        assert Cloud.AWS == "aws"
        assert Cloud.AZURE == "azure"
        assert Cloud.GCP == "gcp"
        assert Cloud.LAMBDA == "lambda"

    def test_cloud_enum_string_conversion(self):
        """Test string conversion of enum values."""
        assert str(Cloud.AWS) == "aws"
        assert str(Cloud.GCP) == "gcp"

    def test_cloud_enum_membership(self):
        """Test membership checks."""
        assert "aws" in [cloud.value for cloud in Cloud]
        assert "invalid_cloud" not in [cloud.value for cloud in Cloud]


class TestListInstancesRequest:
    """Test cases for the ListInstancesRequest model."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        request = ListInstancesRequest()

        assert request.gpus_only is True
        assert request.name_filter is None
        assert request.region_filter is None
        assert request.quantity_filter is None
        assert request.clouds is None
        assert request.case_sensitive is True
        assert request.all_regions is False
        assert request.drop_instance_type_none is True

    def test_valid_request_with_all_fields(self):
        """Test creating a request with all valid fields."""
        request = ListInstancesRequest(
            gpus_only=False,
            name_filter="test-.*",
            region_filter="us-.*",
            quantity_filter=8,
            clouds=["aws", "gcp"],
            case_sensitive=False,
            all_regions=True,
            drop_instance_type_none=False,
        )

        assert request.gpus_only is False
        assert request.name_filter == "test-.*"
        assert request.region_filter == "us-.*"
        assert request.quantity_filter == 8
        assert request.clouds == ["aws", "gcp"]
        assert request.case_sensitive is False
        assert request.all_regions is True
        assert request.drop_instance_type_none is False

    def test_clouds_as_single_string(self):
        """Test clouds field with a single string value."""
        request = ListInstancesRequest(clouds="aws")
        assert request.clouds == "aws"

    def test_clouds_as_list(self):
        """Test clouds field with a list of strings."""
        cloud_list = ["aws", "gcp", "azure"]
        request = ListInstancesRequest(clouds=cloud_list)
        assert request.clouds == cloud_list

    def test_clouds_as_none(self):
        """Test clouds field as None."""
        request = ListInstancesRequest(clouds=None)
        assert request.clouds is None

    def test_quantity_filter_validation(self):
        """Test quantity_filter field validation."""
        # Valid positive integer
        request = ListInstancesRequest(quantity_filter=4)
        assert request.quantity_filter == 4

        # Zero should be valid
        request = ListInstancesRequest(quantity_filter=0)
        assert request.quantity_filter == 0

    def test_boolean_fields(self):
        """Test boolean field validation."""
        request = ListInstancesRequest(
            gpus_only=False,
            case_sensitive=False,
            all_regions=True,
            drop_instance_type_none=False,
        )

        assert request.gpus_only is False
        assert request.case_sensitive is False
        assert request.all_regions is True
        assert request.drop_instance_type_none is False


class TestInstanceTypeInfo:
    """Test cases for the InstanceTypeInfo model."""

    def test_minimal_valid_instance(self):
        """Test creating an instance with only required fields."""
        instance = InstanceTypeInfo(  # type: ignore
            cloud="aws",
            accelerator_name="V100",
            accelerator_count=1.0,
            region="us-east-1",
        )

        assert instance.cloud == "aws"
        assert instance.instance_type is None
        assert instance.accelerator_name == "V100"
        assert instance.accelerator_count == 1.0
        assert instance.cpu_count is None
        assert instance.device_memory is None
        assert instance.memory is None
        assert instance.price is None
        assert instance.spot_price is None
        assert instance.region == "us-east-1"

    def test_complete_valid_instance(self):
        """Test creating an instance with all fields."""
        instance = InstanceTypeInfo(
            cloud="gcp",
            instance_type="n1-standard-4",
            accelerator_name="T4",
            accelerator_count=2.0,
            cpu_count=4.0,
            device_memory=16.0,
            memory=15.0,
            price=0.50,
            spot_price=0.15,
            region="us-central1-a",
        )

        assert instance.cloud == "gcp"
        assert instance.instance_type == "n1-standard-4"
        assert instance.accelerator_name == "T4"
        assert instance.accelerator_count == 2.0
        assert instance.cpu_count == 4.0
        assert instance.device_memory == 16.0
        assert instance.memory == 15.0
        assert instance.price == 0.50
        assert instance.spot_price == 0.15
        assert instance.region == "us-central1-a"

    def test_missing_required_fields(self):
        """Test that ValidationError is raised for missing required fields."""
        # Missing cloud
        with pytest.raises(ValidationError) as exc_info:
            InstanceTypeInfo(  # type: ignore
                accelerator_name="V100", accelerator_count=1.0, region="us-east-1"
            )
        assert "cloud" in str(exc_info.value)

        # Missing accelerator_name
        with pytest.raises(ValidationError) as exc_info:
            InstanceTypeInfo(  # type: ignore
                cloud="aws", accelerator_count=1.0, region="us-east-1"
            )
        assert "accelerator_name" in str(exc_info.value)

        # Missing accelerator_count
        with pytest.raises(ValidationError) as exc_info:
            InstanceTypeInfo(  # type: ignore
                cloud="aws", accelerator_name="V100", region="us-east-1"
            )
        assert "accelerator_count" in str(exc_info.value)

        # Missing region
        with pytest.raises(ValidationError) as exc_info:
            InstanceTypeInfo(  # type: ignore
                cloud="aws", accelerator_name="V100", accelerator_count=1.0
            )
        assert "region" in str(exc_info.value)

    def test_numeric_fields_validation(self):
        """Test numeric field validation."""
        # Valid float values
        instance = InstanceTypeInfo(  # type: ignore
            cloud="aws",
            accelerator_name="V100",
            accelerator_count=1.5,
            cpu_count=2.5,
            device_memory=11.5,
            memory=7.5,
            price=0.123,
            spot_price=0.456,
            region="us-east-1",
        )

        assert instance.accelerator_count == 1.5
        assert instance.cpu_count == 2.5
        assert instance.device_memory == 11.5
        assert instance.memory == 7.5
        assert instance.price == 0.123
        assert instance.spot_price == 0.456

    def test_zero_values(self):
        """Test handling of zero values."""
        instance = InstanceTypeInfo(  # type: ignore
            cloud="aws",
            accelerator_name="CPU",
            accelerator_count=0.0,
            cpu_count=0.0,
            device_memory=0.0,
            memory=0.0,
            price=0.0,
            spot_price=0.0,
            region="us-east-1",
        )

        assert instance.accelerator_count == 0.0
        assert instance.cpu_count == 0.0
        assert instance.device_memory == 0.0
        assert instance.memory == 0.0
        assert instance.price == 0.0
        assert instance.spot_price == 0.0


class TestCloudFilterType:
    """Test cases for the CloudFilter type alias."""

    def test_cloud_filter_none(self):
        """Test CloudFilter with None value."""
        cloud_filter: CloudFilter = None
        assert cloud_filter is None

    def test_cloud_filter_single_string(self):
        """Test CloudFilter with single string."""
        cloud_filter: CloudFilter = "aws"
        assert cloud_filter == "aws"

    def test_cloud_filter_list_of_strings(self):
        """Test CloudFilter with list of strings."""
        cloud_filter: CloudFilter = ["aws", "gcp", "azure"]
        assert cloud_filter == ["aws", "gcp", "azure"]

    def test_cloud_filter_empty_list(self):
        """Test CloudFilter with empty list."""
        cloud_filter: CloudFilter = []
        assert cloud_filter == []

    def test_cloud_filter_in_list_instances_request(self):
        """Test CloudFilter usage in ListInstancesRequest."""
        # Test with None
        request = ListInstancesRequest(clouds=None)
        assert request.clouds is None

        # Test with string
        request = ListInstancesRequest(clouds="aws")
        assert request.clouds == "aws"

        # Test with list
        request = ListInstancesRequest(clouds=["aws", "gcp"])
        assert request.clouds == ["aws", "gcp"]
