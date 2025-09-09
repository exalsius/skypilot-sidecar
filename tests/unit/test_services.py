import json
import os

import pandas as pd
import pytest

from src.models import Cloud, InstanceTypeInfo
from src.services import (
    _clean_concat_dataframes,
    _load_instances_of_cloud,
    list_instances_stream,
    load_all_instances,
    read_catalog_file,
)


@pytest.fixture(autouse=True)
def clean_environment():
    """Ensure each test starts with original environment state."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return """InstanceType,AcceleratorName,AcceleratorCount,vCPUs,MemoryGiB,GpuInfo,Price,SpotPrice,Region,AvailabilityZone
g4ad.2xlarge,Radeon Pro V520,1.0,8.0,32.0,"{'Gpus': [{'Name': 'Radeon Pro V520', 'Manufacturer': 'AMD', 'Count': 1, 'MemoryInfo': {'SizeInMiB': 8192}}], 'TotalGpuMemoryInMiB': 8192}",0.60421,0.244000,us-east-1,use1-az1
p3.2xlarge,V100,1.0,8.0,61.0,"{'Gpus': [{'Name': 'V100', 'Manufacturer': 'NVIDIA', 'Count': 1, 'MemoryInfo': {'SizeInMiB': 16384}}], 'TotalGpuMemoryInMiB': 16384}",3.06000,0.918000,us-east-1,use1-az1
m5.large,CPU-ONLY,0.0,2.0,8.0,,0.096000,0.0288000,us-east-1,use1-az1
t2.micro,,0.0,1.0,1.0,,0.0116000,0.0035000,us-east-1,use1-az1"""


@pytest.fixture
def sample_csv_file(sample_csv_data, tmp_path):
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "vms.csv"
    csv_file.write_text(sample_csv_data)
    return csv_file


@pytest.fixture
def empty_csv_file(tmp_path):
    """Create an empty CSV file for testing."""
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text(
        "InstanceType,AcceleratorName,AcceleratorCount,vCPUs,MemoryGiB,GpuInfo,Price,SpotPrice,Region,AvailabilityZone\n"
    )
    return csv_file


@pytest.fixture
def sample_dataframes():
    """Sample dataframes for testing concatenation."""
    df1 = pd.DataFrame(
        {
            "InstanceType": ["m5.large", "p3.2xlarge"],
            "AcceleratorName": ["CPU-ONLY", "V100"],
            "AcceleratorCount": [0.0, 1.0],
            "vCPUs": [2.0, 8.0],
            "MemoryGiB": [8.0, 61.0],
            "Price": [0.096, 3.06],
            "SpotPrice": [0.0288, 0.918],
            "Region": ["us-east-1", "us-east-1"],
        }
    )

    df2 = pd.DataFrame(
        {
            "InstanceType": ["g4ad.2xlarge"],
            "AcceleratorName": ["Radeon Pro V520"],
            "AcceleratorCount": [1.0],
            "vCPUs": [8.0],
            "MemoryGiB": [32.0],
            "DeviceMemoryGiB": [8.0],
            "Price": [0.60421],
            "SpotPrice": [0.244],
            "Region": ["us-east-1"],
        }
    )

    return [df1, df2]


class TestReadCatalogFile:
    """Tests for read_catalog_file function."""

    def test_read_existing_file(self, sample_csv_file):
        """Test reading an existing CSV file."""
        df = read_catalog_file(sample_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert list(df.columns) == [
            "InstanceType",
            "AcceleratorName",
            "AcceleratorCount",
            "vCPUs",
            "MemoryGiB",
            "GpuInfo",
            "Price",
            "SpotPrice",
            "Region",
            "AvailabilityZone",
        ]
        assert df.iloc[0]["InstanceType"] == "g4ad.2xlarge"

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading a non-existent file raises AssertionError."""
        nonexistent_file = tmp_path / "nonexistent.csv"

        with pytest.raises(AssertionError, match="Catalog file .* does not exist"):
            read_catalog_file(nonexistent_file)


class TestLoadInstancesOfCloud:
    """Tests for _load_instances_of_cloud function."""

    def test_load_existing_file_basic(self, sample_csv_file):
        """Test loading instances from existing file with basic parameters."""
        df = _load_instances_of_cloud(
            catalog_file_path=sample_csv_file,
            gpus_only=False,
            name_filter=None,
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            drop_instance_type_none=True,
        )

        assert df is not None
        assert len(df) == 4
        # Check that AcceleratorName was filled
        assert "CPU-ONLY" in df["AcceleratorName"].values
        # Check that columns are properly selected
        expected_columns = [
            "InstanceType",
            "AcceleratorName",
            "AcceleratorCount",
            "vCPUs",
            "DeviceMemoryGiB",
            "MemoryGiB",
            "Price",
            "SpotPrice",
            "Region",
        ]
        assert list(df.columns) == expected_columns

    def test_load_gpus_only_filter(self, sample_csv_file):
        """Test loading with gpus_only filter."""
        df = _load_instances_of_cloud(
            catalog_file_path=sample_csv_file,
            gpus_only=True,
            name_filter=None,
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            drop_instance_type_none=True,
        )

        assert df is not None
        assert len(df) == 2  # Only GPU instances
        assert all(df["AcceleratorName"] != "CPU-ONLY")

    def test_load_with_name_filter(self, sample_csv_file):
        """Test loading with name filter."""
        df = _load_instances_of_cloud(
            catalog_file_path=sample_csv_file,
            gpus_only=False,
            name_filter="V100",
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            drop_instance_type_none=True,
        )

        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["AcceleratorName"] == "V100"

    def test_load_with_region_filter(self, sample_csv_file):
        """Test loading with region filter."""
        df = _load_instances_of_cloud(
            catalog_file_path=sample_csv_file,
            gpus_only=False,
            name_filter=None,
            region_filter="us-east-1",
            quantity_filter=None,
            case_sensitive=True,
            drop_instance_type_none=True,
        )

        assert df is not None
        assert len(df) == 4
        assert all(df["Region"] == "us-east-1")

    def test_load_with_quantity_filter(self, sample_csv_file):
        """Test loading with quantity filter."""
        df = _load_instances_of_cloud(
            catalog_file_path=sample_csv_file,
            gpus_only=False,
            name_filter=None,
            region_filter=None,
            quantity_filter=1,
            case_sensitive=True,
            drop_instance_type_none=True,
        )

        assert df is not None
        assert len(df) == 2  # Only instances with 1 accelerator
        assert all(df["AcceleratorCount"] == 1.0)

    def test_load_with_drop_instance_type_none_false(self, tmp_path):
        """Test loading with drop_instance_type_none=False."""
        # Create CSV with None instance type
        csv_data = """InstanceType,AcceleratorName,AcceleratorCount,vCPUs,MemoryGiB,GpuInfo,Price,SpotPrice,Region,AvailabilityZone
,V100,1.0,8.0,61.0,"{'Gpus': [{'Name': 'V100', 'Manufacturer': 'NVIDIA', 'Count': 1, 'MemoryInfo': {'SizeInMiB': 16384}}], 'TotalGpuMemoryInMiB': 16384}",3.06000,0.918000,us-east-1,use1-az1
p3.2xlarge,V100,1.0,8.0,61.0,"{'Gpus': [{'Name': 'V100', 'Manufacturer': 'NVIDIA', 'Count': 1, 'MemoryInfo': {'SizeInMiB': 16384}}], 'TotalGpuMemoryInMiB': 16384}",3.06000,0.918000,us-east-1,use1-az1"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_data)

        df = _load_instances_of_cloud(
            catalog_file_path=csv_file,
            gpus_only=False,
            name_filter=None,
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            drop_instance_type_none=False,
        )

        assert df is not None
        assert len(df) == 2  # Both instances included

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from non-existent file returns None."""
        nonexistent_file = tmp_path / "nonexistent.csv"

        df = _load_instances_of_cloud(
            catalog_file_path=nonexistent_file,
            gpus_only=False,
            name_filter=None,
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            drop_instance_type_none=True,
        )

        assert df is None

    def test_load_directory_instead_of_file(self, tmp_path):
        """Test loading from directory instead of file returns None."""
        df = _load_instances_of_cloud(
            catalog_file_path=tmp_path,
            gpus_only=False,
            name_filter=None,
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            drop_instance_type_none=True,
        )

        assert df is None

    def test_device_memory_calculation(self, sample_csv_file):
        """Test that device memory is calculated correctly from GpuInfo."""
        df = _load_instances_of_cloud(
            catalog_file_path=sample_csv_file,
            gpus_only=True,
            name_filter=None,
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            drop_instance_type_none=True,
        )

        assert df is not None
        # Check V100 has 16 GiB device memory (16384 MiB / 1024)
        v100_row = df[df["AcceleratorName"] == "V100"].iloc[0]
        assert v100_row["DeviceMemoryGiB"] == 16.0

        # Check Radeon Pro V520 has 8 GiB device memory (8192 MiB / 1024)
        radeon_row = df[df["AcceleratorName"] == "Radeon Pro V520"].iloc[0]
        assert radeon_row["DeviceMemoryGiB"] == 8.0

    def test_invalid_gpu_info(self, tmp_path):
        """Test handling of invalid GpuInfo data."""
        csv_data = """InstanceType,AcceleratorName,AcceleratorCount,vCPUs,MemoryGiB,GpuInfo,Price,SpotPrice,Region,AvailabilityZone
p3.2xlarge,V100,1.0,8.0,61.0,invalid_json,3.06000,0.918000,us-east-1,use1-az1"""
        csv_file = tmp_path / "invalid_gpu.csv"
        csv_file.write_text(csv_data)

        df = _load_instances_of_cloud(
            catalog_file_path=csv_file,
            gpus_only=False,
            name_filter=None,
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            drop_instance_type_none=True,
        )

        assert df is not None
        assert len(df) == 1
        # DeviceMemoryGiB should be None due to invalid GpuInfo
        assert pd.isna(df.iloc[0]["DeviceMemoryGiB"])


class TestCleanConcatDataframes:
    """Tests for _clean_concat_dataframes function."""

    def test_concat_valid_dataframes(self, sample_dataframes):
        """Test concatenating valid dataframes."""
        result = _clean_concat_dataframes(sample_dataframes)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 2 + 1 rows
        # Should have union of all columns
        expected_columns = [
            "AcceleratorCount",
            "AcceleratorName",
            "DeviceMemoryGiB",
            "InstanceType",
            "MemoryGiB",
            "Price",
            "Region",
            "SpotPrice",
            "vCPUs",
        ]
        assert sorted(result.columns.tolist()) == expected_columns

    def test_concat_with_none_values(self, sample_dataframes):
        """Test concatenating with None values in list."""
        dataframes_with_none = sample_dataframes + [None, None]
        result = _clean_concat_dataframes(dataframes_with_none)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Same as without None values

    def test_concat_empty_dataframes(self):
        """Test concatenating empty dataframes."""
        empty_df1 = pd.DataFrame()
        empty_df2 = pd.DataFrame({"col1": [], "col2": []})

        result = _clean_concat_dataframes([empty_df1, empty_df2])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_concat_all_na_dataframes(self):
        """Test concatenating dataframes with all NA values."""
        na_df = pd.DataFrame({"col1": [None, None], "col2": [None, None]})

        result = _clean_concat_dataframes([na_df])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_concat_mixed_columns(self):
        """Test concatenating dataframes with different columns."""
        df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df2 = pd.DataFrame({"B": [5], "C": [6]})

        result = _clean_concat_dataframes([df1, df2])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert sorted(result.columns.tolist()) == ["A", "B", "C"]


class TestLoadAllInstances:
    """Tests for load_all_instances function."""

    def test_load_single_cloud(self, tmp_path, sample_csv_data):
        """Test loading instances from a single cloud."""
        # Create directory structure
        aws_dir = tmp_path / "aws"
        aws_dir.mkdir()
        csv_file = aws_dir / "vms.csv"
        csv_file.write_text(sample_csv_data)

        result = load_all_instances(
            catalog_root_directory=tmp_path,
            catalog_filename="vms.csv",
            clouds=[Cloud.AWS],
            gpus_only=False,
            name_filter=None,
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            all_regions=False,
            drop_instance_type_none=True,
        )

        assert isinstance(result, dict)
        assert len(result) == 3  # CPU-ONLY, Radeon Pro V520, V100
        assert "CPU-ONLY" in result
        assert "Radeon Pro V520" in result
        assert "V100" in result

        # Check instance info structure
        v100_instances = result["V100"]
        assert len(v100_instances) == 1
        instance = v100_instances[0]
        assert isinstance(instance, InstanceTypeInfo)
        assert instance.cloud == "aws"
        assert instance.instance_type == "p3.2xlarge"
        assert instance.accelerator_name == "V100"

    def test_load_multiple_clouds(self, tmp_path, sample_csv_data):
        """Test loading instances from multiple clouds."""
        # Create directory structure for multiple clouds
        for cloud in ["aws", "gcp"]:
            cloud_dir = tmp_path / cloud
            cloud_dir.mkdir()
            csv_file = cloud_dir / "vms.csv"
            csv_file.write_text(sample_csv_data)

        result = load_all_instances(
            catalog_root_directory=tmp_path,
            catalog_filename="vms.csv",
            clouds=[Cloud.AWS, Cloud.GCP],
            gpus_only=False,
            name_filter=None,
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            all_regions=False,
            drop_instance_type_none=True,
        )

        assert isinstance(result, dict)
        # Each accelerator type should have instances from both clouds
        v100_instances = result.get("V100", [])
        clouds_represented = {instance.cloud for instance in v100_instances}
        assert "aws" in clouds_represented
        assert "gcp" in clouds_represented

    def test_load_with_all_regions_true(self, tmp_path, sample_csv_data):
        """Test loading with all_regions=True includes region in sorting."""
        aws_dir = tmp_path / "aws"
        aws_dir.mkdir()
        csv_file = aws_dir / "vms.csv"
        csv_file.write_text(sample_csv_data)

        result = load_all_instances(
            catalog_root_directory=tmp_path,
            catalog_filename="vms.csv",
            clouds=[Cloud.AWS],
            gpus_only=False,
            name_filter=None,
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            all_regions=True,
            drop_instance_type_none=True,
        )

        assert isinstance(result, dict)
        # Should still have same accelerator types
        assert len(result) == 3

    def test_load_nonexistent_cloud(self, tmp_path):
        """Test loading from non-existent cloud directory."""
        result = load_all_instances(
            catalog_root_directory=tmp_path,
            catalog_filename="vms.csv",
            clouds=[Cloud.AWS],
            gpus_only=False,
            name_filter=None,
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            all_regions=False,
            drop_instance_type_none=True,
        )

        assert isinstance(result, dict)
        assert len(result) == 0  # No instances loaded

    def test_instance_sorting(self, tmp_path):
        """Test that instances are properly sorted."""
        # Create CSV with multiple instances for sorting test
        csv_data = """InstanceType,AcceleratorName,AcceleratorCount,vCPUs,MemoryGiB,GpuInfo,Price,SpotPrice,Region,AvailabilityZone
p3.2xlarge,V100,1.0,8.0,61.0,"{'Gpus': [{'Name': 'V100', 'Manufacturer': 'NVIDIA', 'Count': 1, 'MemoryInfo': {'SizeInMiB': 16384}}], 'TotalGpuMemoryInMiB': 16384}",3.06000,0.918000,us-west-2,usw2-az1
p3.2xlarge,V100,1.0,8.0,61.0,"{'Gpus': [{'Name': 'V100', 'Manufacturer': 'NVIDIA', 'Count': 1, 'MemoryInfo': {'SizeInMiB': 16384}}], 'TotalGpuMemoryInMiB': 16384}",2.50000,0.750000,us-east-1,use1-az1
p3.8xlarge,V100,4.0,32.0,244.0,"{'Gpus': [{'Name': 'V100', 'Manufacturer': 'NVIDIA', 'Count': 4, 'MemoryInfo': {'SizeInMiB': 16384}}], 'TotalGpuMemoryInMiB': 65536}",12.24000,3.672000,us-east-1,use1-az1"""

        aws_dir = tmp_path / "aws"
        aws_dir.mkdir()
        csv_file = aws_dir / "vms.csv"
        csv_file.write_text(csv_data)

        result = load_all_instances(
            catalog_root_directory=tmp_path,
            catalog_filename="vms.csv",
            clouds=[Cloud.AWS],
            gpus_only=True,
            name_filter=None,
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            all_regions=False,
            drop_instance_type_none=True,
        )

        v100_instances = result["V100"]
        assert len(v100_instances) == 3

        # Should be sorted by accelerator_count, instance_type, cpu_count, price, spot_price, region
        assert v100_instances[0].accelerator_count == 1.0  # 1 GPU instances first
        assert v100_instances[2].accelerator_count == 4.0  # 4 GPU instances last

        # Among 1 GPU instances, should be sorted by price
        assert v100_instances[0].price == 2.50000  # Cheaper first
        assert v100_instances[1].price == 3.06000  # More expensive second


@pytest.mark.asyncio
class TestListInstancesStream:
    """Tests for list_instances_stream async function."""

    async def test_stream_instances(self, tmp_path, sample_csv_data):
        """Test streaming instances as JSON."""
        aws_dir = tmp_path / "aws"
        aws_dir.mkdir()
        csv_file = aws_dir / "vms.csv"
        csv_file.write_text(sample_csv_data)

        stream_gen = list_instances_stream(
            catalog_root_directory=tmp_path,
            catalog_filename="vms.csv",
            clouds=[Cloud.AWS],
            gpus_only=False,
            name_filter=None,
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            all_regions=False,
            drop_instance_type_none=True,
        )

        results = []
        async for item in stream_gen:
            results.append(item)

        assert len(results) == 3  # Should have 3 accelerator types

        # Each result should be valid JSON ending with newline
        for result in results:
            assert result.endswith("\n")
            data = json.loads(result.strip())
            assert isinstance(data, dict)
            # Each data should have exactly one key (accelerator name)
            assert len(data) == 1

            # The value should be a list of instance dicts
            accelerator_name, instances = next(iter(data.items()))
            assert isinstance(instances, list)
            assert len(instances) > 0

            # Check instance structure
            instance = instances[0]
            assert "cloud" in instance
            assert "instance_type" in instance
            assert "accelerator_name" in instance
            assert instance["accelerator_name"] == accelerator_name

    async def test_stream_with_filters(self, tmp_path, sample_csv_data):
        """Test streaming with filters applied."""
        aws_dir = tmp_path / "aws"
        aws_dir.mkdir()
        csv_file = aws_dir / "vms.csv"
        csv_file.write_text(sample_csv_data)

        stream_gen = list_instances_stream(
            catalog_root_directory=tmp_path,
            catalog_filename="vms.csv",
            clouds=[Cloud.AWS],
            gpus_only=True,  # Only GPUs
            name_filter="V100",  # Only V100
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            all_regions=False,
            drop_instance_type_none=True,
        )

        results = []
        async for item in stream_gen:
            results.append(item)

        assert len(results) == 1  # Should only have V100

        data = json.loads(results[0].strip())
        assert "V100" in data
        assert len(data["V100"]) == 1

    async def test_stream_empty_result(self, tmp_path):
        """Test streaming when no instances are found."""
        # Create empty directory structure
        aws_dir = tmp_path / "aws"
        aws_dir.mkdir()

        stream_gen = list_instances_stream(
            catalog_root_directory=tmp_path,
            catalog_filename="vms.csv",
            clouds=[Cloud.AWS],
            gpus_only=False,
            name_filter=None,
            region_filter=None,
            quantity_filter=None,
            case_sensitive=True,
            all_regions=False,
            drop_instance_type_none=True,
        )

        results = []
        async for item in stream_gen:
            results.append(item)

        assert len(results) == 0  # No instances to stream
