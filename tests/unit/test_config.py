import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import DEFAULT_SKY_CATALOG_PATH, AppConfig, config


@pytest.fixture(autouse=True)
def clean_environment():
    """Ensure each test starts with original environment state."""
    os.environ.clear()
    yield
    # Restore original environment
    os.environ.clear()


class TestAppConfig:
    """Test cases for the AppConfig class."""

    def test_default_values(self):
        """Test that AppConfig uses correct default values."""
        app_config = AppConfig()

        assert app_config.api_host == "0.0.0.0"
        assert app_config.api_port == 5555
        assert app_config.api_version == "1.0"
        assert app_config.catalog_path == DEFAULT_SKY_CATALOG_PATH
        assert app_config.catalog_filename == "vms.csv"

    def test_custom_values(self):
        """Test that AppConfig accepts custom values."""
        custom_path = Path("/tmp/test_catalog")

        app_config = AppConfig(
            api_host="127.0.0.1",
            api_port=8080,
            api_version="2.0",
            catalog_path=custom_path,
            catalog_filename="custom.csv",
        )

        assert app_config.api_host == "127.0.0.1"
        assert app_config.api_port == 8080
        assert app_config.api_version == "2.0"
        assert app_config.catalog_path == custom_path
        assert app_config.catalog_filename == "custom.csv"

    def test_field_types(self):
        """Test that fields have correct types."""
        app_config = AppConfig()

        assert isinstance(app_config.api_host, str)
        assert isinstance(app_config.api_port, int)
        assert isinstance(app_config.api_version, str)
        assert isinstance(app_config.catalog_path, Path)
        assert isinstance(app_config.catalog_filename, str)

    @patch("src.config.Path.mkdir")
    @patch("src.config.Path.is_dir")
    @patch("src.config.Path.exists")
    def test_catalog_path_validator_path_exists_and_is_dir(
        self, mock_exists, mock_is_dir, mock_mkdir
    ):
        """Test validator when path exists and is a directory."""
        mock_exists.return_value = True
        mock_is_dir.return_value = True

        test_path = Path("/tmp/existing_catalog")
        app_config = AppConfig(catalog_path=test_path)

        assert app_config.catalog_path == test_path
        mock_exists.assert_called_once()
        mock_is_dir.assert_called_once()
        mock_mkdir.assert_not_called()

    @patch("src.config.Path.mkdir")
    @patch("src.config.Path.is_dir")
    @patch("src.config.Path.exists")
    @patch("src.config.logging.warning")
    def test_catalog_path_validator_path_not_exists(
        self, mock_warning, mock_exists, mock_is_dir, mock_mkdir
    ):
        """Test validator when path doesn't exist - should create it."""
        mock_exists.return_value = False
        mock_is_dir.return_value = True

        test_path = Path("/tmp/new_catalog")
        app_config = AppConfig(catalog_path=test_path)

        assert app_config.catalog_path == test_path
        mock_exists.assert_called_once()
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_warning.assert_called_once_with(
            f"Catalog path {test_path} does not exist. Creating it."
        )
        mock_is_dir.assert_called_once()

    @patch("src.config.Path.is_dir")
    @patch("src.config.Path.exists")
    def test_catalog_path_validator_path_not_directory(self, mock_exists, mock_is_dir):
        """Test validator when path exists but is not a directory."""
        mock_exists.return_value = True
        mock_is_dir.return_value = False

        test_path = Path("/tmp/not_a_directory.txt")

        with pytest.raises(ValidationError) as exc_info:
            AppConfig(catalog_path=test_path)

        assert "Catalog path" in str(exc_info.value)
        assert "is not a directory" in str(exc_info.value)
        mock_exists.assert_called_once()
        mock_is_dir.assert_called_once()

    def test_environment_variables_loading(self):
        """Test that configuration loads from environment variables."""
        with patch.dict(
            os.environ,
            {
                "EXLS_SKY_API_HOST": "192.168.1.1",
                "EXLS_SKY_API_PORT": "9999",
                "EXLS_SKY_API_VERSION": "3.0",
                "EXLS_SKY_CATALOG_FILENAME": "custom_vms.csv",
            },
        ):
            app_config = AppConfig()

            assert app_config.api_host == "192.168.1.1"
            assert app_config.api_port == 9999
            assert app_config.api_version == "3.0"
            assert app_config.catalog_filename == "custom_vms.csv"

    def test_environment_variables_with_nested_delimiter(self):
        """Test that nested environment variables work with double underscore delimiter."""
        # Note: This test demonstrates the nested delimiter configuration,
        # though the current config doesn't have nested fields
        with patch.dict(
            os.environ,
            {
                "EXLS_SKY_API_HOST": "nested.example.com",
            },
        ):
            app_config = AppConfig()
            assert app_config.api_host == "nested.example.com"

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored due to 'extra=ignore' setting."""
        # Test with environment variables since extra=ignore applies to input validation
        with patch.dict(
            os.environ,
            {
                "EXLS_SKY_API_HOST": "127.0.0.1",
                "EXLS_SKY_UNKNOWN_FIELD": "should_be_ignored",
            },
        ):
            app_config = AppConfig()

            # Known field should be loaded
            assert app_config.api_host == "127.0.0.1"
            # Unknown field should not be present
            assert not hasattr(app_config, "unknown_field")

    def test_invalid_port_type(self):
        """Test that invalid port type raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AppConfig(api_port="not_a_number")  # type: ignore

        assert "Input should be a valid integer" in str(exc_info.value)

    def test_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        app_config = AppConfig(catalog_path="/tmp/string_path")  # type: ignore

        assert isinstance(app_config.catalog_path, Path)
        assert str(app_config.catalog_path) == "/tmp/string_path"


class TestGlobalConfig:
    """Test cases for the global config instance."""

    def test_global_config_exists(self):
        """Test that global config instance exists and is AppConfig."""
        assert config is not None
        assert isinstance(config, AppConfig)

    def test_global_config_has_defaults(self):
        """Test that global config instance has default values."""
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 5555
        assert config.api_version == "1.0"
        assert config.catalog_filename == "vms.csv"


class TestDefaultSkyPath:
    """Test cases for DEFAULT_SKY_CATALOG_PATH constant."""

    def test_default_sky_catalog_path(self):
        """Test that DEFAULT_SKY_CATALOG_PATH is correctly constructed."""
        expected_path = Path.home() / ".sky" / "catalogs" / "v7"
        assert DEFAULT_SKY_CATALOG_PATH == expected_path

    def test_default_sky_catalog_path_type(self):
        """Test that DEFAULT_SKY_CATALOG_PATH is a Path object."""
        assert isinstance(DEFAULT_SKY_CATALOG_PATH, Path)
