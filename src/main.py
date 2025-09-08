import ast
import json
from enum import StrEnum
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union, cast

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

CloudFilter = Optional[Union[List[str], str]]

DEFAULT_SKY_CATALOG_PATH: Path = Path.home() / ".sky" / "catalogs" / "v7"


class Cloud(StrEnum):
    AWS = "aws"
    AZURE = "azure"
    CUDO = "cudo"
    DO = "do"
    FLUIDSTACK = "fluidstack"
    GCP = "gcp"
    HYPERBOLIC = "hyperbolic"
    IBM = "ibm"
    KUBERNETES = "kubernetes"
    LAMBDA = "lambda"
    NEBIUS = "nebius"
    OCI = "oci"
    PAPERSPACE = "paperspace"
    RUNPOD = "runpod"
    SCP = "scp"
    VAST = "vast"
    VSPHERE = "vsphere"


SKYPILOT_ENV_VAR_PREFIX = "SKYPILOT_"
SKY_API_SERVER_URL_ENV_VAR = f"{SKYPILOT_ENV_VAR_PREFIX}API_SERVER_ENDPOINT"


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
            raise ValueError(f"Catalog path {v} does not exist.")
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


class ListInstancesRequest(BaseModel):
    gpus_only: bool = Field(True, description="Whether to only show GPU instances.")
    name_filter: Optional[str] = Field(
        None, description="A regex to filter by accelerator name."
    )
    region_filter: Optional[str] = Field(
        None, description="A regex to filter by region."
    )
    quantity_filter: Optional[int] = Field(
        None, description="A regex to filter by quantity."
    )
    clouds: CloudFilter = Field(None, description="A list of clouds to filter by.")
    case_sensitive: bool = Field(
        True, description="Whether the regex is case-sensitive."
    )
    all_regions: bool = Field(False, description="Whether to show all regions.")
    drop_instance_type_none: bool = Field(
        True, description="Whether to drop instance types that are None."
    )


class InstanceTypeInfo(BaseModel):
    """Instance type information.

    - cloud: Cloud name.
    - instance_type: String that can be used in YAML to specify this instance
      type. E.g. `p3.2xlarge`.
    - accelerator_name: Canonical name of the accelerator. E.g. `V100`.
    - accelerator_count: Number of accelerators offered by this instance type.
    - cpu_count: Number of vCPUs offered by this instance type.
    - device_memory: Device memory in GiB.
    - memory: Instance memory in GiB.
    - price: Regular instance price per hour (cheapest across all regions).
    - spot_price: Spot instance price per hour (cheapest across all regions).
    - region: Region where this instance type belongs to.
    """

    cloud: str = Field(..., description="Cloud name.")
    instance_type: Optional[str] = Field(
        None,
        description="String that can be used in YAML to specify this instance type. E.g. `p3.2xlarge`.",
    )
    accelerator_name: str = Field(
        ..., description="Canonical name of the accelerator. E.g. `V100`."
    )
    accelerator_count: float = Field(
        ..., description="Number of accelerators offered by this instance type."
    )
    cpu_count: Optional[float] = Field(
        None, description="Number of vCPUs offered by this instance type."
    )
    device_memory: Optional[float] = Field(None, description="Device memory in GiB.")
    memory: Optional[float] = Field(None, description="Instance memory in GiB.")
    price: Optional[float] = Field(
        None,
        description="Regular instance price per hour (cheapest across all regions).",
    )
    spot_price: Optional[float] = Field(
        None,
        description="Spot instance price per hour (cheapest across all regions).",
    )
    region: str = Field(..., description="Region where this instance type belongs to.")

    # A set of all fields in the InstanceTypeInfo model, for quick lookups.
    # NOTE: This is a ClassVar, so it's shared across all instances of the
    # class. This is fine because it's read-only.
    # model_fields_set: ClassVar[set[str]] = {
    #    field for field in __annotations__ if not field.startswith("_")
    # }


def read_catalog(config: AppConfig) -> pd.DataFrame:
    """Reads the catalogs from all the local CSV files."""

    catalogs_files = []
    for cloud in Cloud:
        catalog_path = config.catalog_path / cloud.value / config.catalog_filename
        if not catalog_path.exists():
            # logger.warning(f"Catalog file {catalog_path} does not exist.")
            continue
        catalogs_files.append(catalog_path)

    if not catalogs_files:
        return pd.DataFrame()

    return pd.concat([pd.read_csv(catalog_path) for catalog_path in catalogs_files])


def _list_instances_impl(
    config: AppConfig,
    cloud: Cloud,
    gpus_only: bool,
    name_filter: Optional[str],
    region_filter: Optional[str],
    quantity_filter: Optional[int],
    case_sensitive: bool,
    all_regions: bool,
    drop_instance_type_none: bool,
) -> Dict[str, List[InstanceTypeInfo]]:
    """Lists accelerators offered in a cloud service catalog.

    `name_filter` is a regular expression used to filter accelerator names
    using pandas.Series.str.contains.

    Returns a mapping from the canonical names of accelerators to a list of
    instance types offered by this cloud.
    """
    df: pd.DataFrame = read_catalog(config)

    df["AcceleratorName"] = df["AcceleratorName"].fillna("CPU-ONLY")

    if gpus_only:
        df = df[df["GpuInfo"].notna()]  # type: ignore
    df = df.copy()  # avoid column assignment warning

    # accelerator_count can be NaN, so we need to handle it.
    df["AcceleratorCount"] = cast(
        pd.Series, pd.to_numeric(df["AcceleratorCount"], errors="coerce")
    ).fillna(0.0)

    try:
        gpu_info_df = df["GpuInfo"].apply(ast.literal_eval)
        df["DeviceMemoryGiB"] = (
            gpu_info_df.apply(lambda row: row["Gpus"][0]["MemoryInfo"]["SizeInMiB"])
            / 1024.0
        )
    except (ValueError, SyntaxError):
        df["DeviceMemoryGiB"] = None

    df = df[  # type: ignore
        [
            "InstanceType",
            "AcceleratorName",
            "AcceleratorCount",
            "vCPUs",
            "DeviceMemoryGiB",  # device memory
            "MemoryGiB",  # host memory
            "Price",
            "SpotPrice",
            "Region",
        ]
    ].drop_duplicates()

    if drop_instance_type_none:
        df = df[df["InstanceType"].notna()]  # type: ignore

    if name_filter is not None:
        df = df[  # type: ignore
            df["AcceleratorName"].str.contains(
                name_filter, case=case_sensitive, regex=True
            )
        ]
    if region_filter is not None:
        df = df[  # type: ignore
            df["Region"].str.contains(region_filter, case=case_sensitive, regex=True)
        ]
    df["AcceleratorCount"] = df["AcceleratorCount"].astype(float)
    if quantity_filter is not None:
        df = df[df["AcceleratorCount"] == quantity_filter]  # type: ignore

    grouped = df.groupby("AcceleratorName")

    def make_list_from_df(rows: pd.DataFrame) -> List[InstanceTypeInfo]:
        sort_key = ["Price", "SpotPrice"]
        subset = [
            "InstanceType",
            "AcceleratorName",
            "AcceleratorCount",
            "vCPUs",
            "MemoryGiB",
        ]
        if all_regions:
            sort_key.append("Region")
            subset.append("Region")

        rows = rows.sort_values(by=sort_key).drop_duplicates(
            subset=subset, keep="first"
        )
        rows = rows.astype(object).where(pd.notna(rows), None)
        ret = rows.apply(
            lambda row: InstanceTypeInfo(
                cloud=cloud.value,
                instance_type=row["InstanceType"],
                accelerator_name=row["AcceleratorName"],
                accelerator_count=row["AcceleratorCount"],
                cpu_count=row["vCPUs"],
                device_memory=row["DeviceMemoryGiB"],
                memory=row["MemoryGiB"],
                price=row["Price"],
                spot_price=row["SpotPrice"],
                region=row["Region"],
            ),
            axis="columns",
        ).tolist()
        # Sort by price and region as well.
        ret.sort(
            key=lambda info: (
                info.accelerator_count,
                info.instance_type or "",
                info.cpu_count if not pd.isna(info.cpu_count) else 0,
                info.price if not pd.isna(info.price) else float("inf"),
                info.spot_price if not pd.isna(info.spot_price) else float("inf"),
                info.region,
            )
        )
        return ret

    return {str(k): make_list_from_df(v) for k, v in grouped}


app = FastAPI()


async def list_instances_stream(
    gpus_only: bool = True,
    name_filter: Optional[str] = None,
    region_filter: Optional[str] = None,
    quantity_filter: Optional[int] = None,
    clouds: CloudFilter = None,
    case_sensitive: bool = True,
    all_regions: bool = False,
    drop_instance_type_none: bool = True,
) -> AsyncGenerator[str, None]:
    config: AppConfig = AppConfig()
    cloud_list: List[Cloud] = []

    if clouds is None:
        cloud_list = [cloud for cloud in Cloud]
    else:
        cloud_list_str: List[str] = [clouds] if isinstance(clouds, str) else clouds
        cloud_list = [Cloud(cloud) for cloud in cloud_list_str if cloud in Cloud]

    for cloud in cloud_list:
        results: Dict[str, List[InstanceTypeInfo]] = _list_instances_impl(
            config,
            cloud,
            gpus_only,
            name_filter,
            region_filter,
            quantity_filter,
            case_sensitive,
            all_regions,
            drop_instance_type_none,
        )

        for gpu_type, items in results.items():
            items_as_dict = [item.model_dump() for item in items]
            yield json.dumps({gpu_type: items_as_dict}) + "\n"
            # for item in items:
            #    yield json.dumps(item.model_dump()) + "\n"


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/list-instances")
async def list_instances_endpoint(body: ListInstancesRequest) -> StreamingResponse:
    return StreamingResponse(
        list_instances_stream(
            gpus_only=body.gpus_only,
            name_filter=body.name_filter,
            region_filter=body.region_filter,
            quantity_filter=body.quantity_filter,
            clouds=body.clouds,
            case_sensitive=body.case_sensitive,
            all_regions=body.all_regions,
            drop_instance_type_none=body.drop_instance_type_none,
        )
    )


if __name__ == "__main__":
    config: AppConfig = AppConfig()
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
    )
