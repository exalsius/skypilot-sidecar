from typing import AsyncGenerator, Callable, List

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.config import config
from src.models import Cloud, ListInstancesRequest
from src.services import list_instances_stream


async def list_instances_stream_wrapper(
    body: ListInstancesRequest,
) -> AsyncGenerator[str, None]:
    cloud_list: List[Cloud] = []

    if body.clouds is None:
        cloud_list = [cloud for cloud in Cloud]
    else:
        cloud_list_str: List[str] = (
            [body.clouds] if isinstance(body.clouds, str) else body.clouds
        )
        cloud_list = [Cloud(cloud) for cloud in cloud_list_str if cloud in Cloud]

    async for item in list_instances_stream(
        catalog_root_directory=config.catalog_path,
        catalog_filename=config.catalog_filename,
        clouds=cloud_list,
        gpus_only=body.gpus_only,
        name_filter=body.name_filter,
        region_filter=body.region_filter,
        quantity_filter=body.quantity_filter,
        case_sensitive=body.case_sensitive,
        all_regions=body.all_regions,
        drop_instance_type_none=body.drop_instance_type_none,
    ):
        yield item


def get_list_instances_stream_wrapper():
    """Get the list_instances_stream_wrapper function."""
    return list_instances_stream_wrapper


router = APIRouter()


@router.post("/list-instances")
async def list_instances_endpoint(
    body: ListInstancesRequest,
    wrapper: Callable = Depends(get_list_instances_stream_wrapper),
) -> StreamingResponse:
    return StreamingResponse(wrapper(body))
