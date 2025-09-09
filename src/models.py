from enum import StrEnum
from typing import List, Optional, Union

from pydantic import BaseModel, Field

CloudFilter = Optional[Union[List[str], str]]


class Cloud(StrEnum):
    AWS = "aws"
    AZURE = "azure"
    CUDO = "cudo"
    DO = "do"
    FLUIDSTACK = "fluidstack"
    GCP = "gcp"
    HYPERBOLIC = "hyperbolic"
    IBM = "ibm"
    LAMBDA = "lambda"
    NEBIUS = "nebius"
    OCI = "oci"
    PAPERSPACE = "paperspace"
    RUNPOD = "runpod"
    SCP = "scp"
    VAST = "vast"


class ListInstancesRequest(BaseModel):
    gpus_only: bool = Field(
        default=True, description="Whether to show only instances with GPUs."
    )
    name_filter: Optional[str] = Field(
        default=None, description="A regex to filter by instance name."
    )
    region_filter: Optional[str] = Field(
        default=None, description="A regex to filter by region."
    )
    quantity_filter: Optional[int] = Field(
        default=None, description="Filter by the number of accelerators."
    )
    clouds: Optional[CloudFilter] = Field(
        default=None,
        description="A list of clouds to query. If not specified, all clouds are queried.",
    )
    case_sensitive: bool = Field(
        default=True, description="Whether the name filter is case-sensitive."
    )
    all_regions: bool = Field(
        default=False, description="Show instances from all regions."
    )
    drop_instance_type_none: bool = Field(
        default=True,
        description="Drop instances where the instance type is not specified.",
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
