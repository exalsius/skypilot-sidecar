import logging
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SKY_CATALOG_PATH: Path = Path.home() / ".sky" / "catalogs" / "v7"


class AppConfig(BaseSettings):
    """App configuration."""

    api_host: str = Field(
        default="0.0.0.0",
        description="Host to run the SkyPilot sidecar API on.",
    )
    api_port: int = Field(
        default=5555,
        description="Port to run the SkyPilot sidecar API on.",
    )
    api_version: str = Field(default="1.0", description="SkyPilot sidecar API version.")

    catalog_path: Path = Field(
        default=DEFAULT_SKY_CATALOG_PATH,
        description="Path to the catalog root directory. Defaults to the user's home directory.",
    )

    @field_validator("catalog_path")
    @classmethod
    def validate_catalog_path(cls, v: Path) -> Path:
        if not v.exists():
            logging.warning(f"Catalog path {v} does not exist. Creating it.")
            v.mkdir(parents=True, exist_ok=True)
        if not v.is_dir():
            raise ValueError(f"Catalog path {v} is not a directory.")
        return v

    catalog_filename: str = Field(
        default="vms.csv",
        description="Name of the catalog file. Defaults to `vms.csv`.",
    )

    model_config = SettingsConfigDict(
        env_prefix="EXLS_SKY_",
        env_nested_delimiter="__",
        extra="ignore",
    )


config = AppConfig()
