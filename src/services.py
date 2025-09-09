import ast
import json
import logging
import re
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, cast

import pandas as pd

from src.models import Cloud, InstanceTypeInfo


def _is_valid_regex(pattern: str) -> bool:
    """Check if a string is a valid regex pattern."""
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def read_catalog_file(catalog_file_path: Path) -> pd.DataFrame:
    """Reads the catalogs from all the local CSV files."""

    assert catalog_file_path.exists(), (
        f"Catalog file {catalog_file_path} does not exist."
    )

    return pd.read_csv(catalog_file_path)


def _filter_by_name(
    df: pd.DataFrame, column_name: str, name_filter: str, case_sensitive: bool
) -> pd.DataFrame:
    """Filter the dataframe by name."""
    if _is_valid_regex(name_filter):
        return df.loc[
            df[column_name].str.contains(
                name_filter, case=case_sensitive, regex=True, na=False
            )
        ]
    else:
        return df.loc[
            df[column_name].str.contains(
                name_filter, case=case_sensitive, regex=False, na=False
            )
        ]


def _load_instances_of_cloud(
    catalog_file_path: Path,
    gpus_only: bool,
    name_filter: Optional[str],
    region_filter: Optional[str],
    quantity_filter: Optional[int],
    case_sensitive: bool,
    drop_instance_type_none: bool,
) -> Optional[pd.DataFrame]:
    """Lists accelerators offered in a cloud service catalog.

    `name_filter` is a regular expression used to filter accelerator names
    using pandas.Series.str.contains.

    Returns a mapping from the canonical names of accelerators to a list of
    instance types offered by this cloud.
    """
    if not catalog_file_path.exists() or not catalog_file_path.is_file():
        logging.warning(f"Catalog file {catalog_file_path} does not exist.")
        return None

    try:
        df: pd.DataFrame = read_catalog_file(catalog_file_path)
    except Exception as e:
        logging.error(f"Error reading catalog file {catalog_file_path}: {e}")
        return None

    df["AcceleratorName"] = df["AcceleratorName"].fillna("CPU-ONLY")

    if gpus_only:
        df = df.loc[df["GpuInfo"].notna()]
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

    df = df.loc[
        :,
        [
            "InstanceType",
            "AcceleratorName",
            "AcceleratorCount",
            "vCPUs",
            "DeviceMemoryGiB",
            "MemoryGiB",
            "Price",
            "SpotPrice",
            "Region",
        ],
    ].drop_duplicates()

    if drop_instance_type_none:
        df = df.loc[df["InstanceType"].notna()]

    if name_filter is not None:
        df = _filter_by_name(df, "AcceleratorName", name_filter, case_sensitive)

    if region_filter is not None:
        df = _filter_by_name(df, "Region", region_filter, case_sensitive)

    df["AcceleratorCount"] = df["AcceleratorCount"].astype(float)
    if quantity_filter is not None:
        df = df.loc[df["AcceleratorCount"] == quantity_filter]

    return df


def _clean_concat_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Clean a list of dataframes before concatenation."""
    # union of columns across all inputs
    all_cols: List[str] = sorted(
        set().union(*[df.columns for df in dfs if df is not None])
    )

    core: List[pd.DataFrame] = [
        df
        for df in dfs
        if df is not None and not df.empty and not df.isna().to_numpy().all()
    ]

    if core:
        # Drop columns that are all NA from each DataFrame before concatenation
        # to avoid a FutureWarning in pandas.
        cleaned_core = [df.dropna(axis=1, how="all") for df in core]
        df_result = pd.concat(cleaned_core, ignore_index=True)
        df_result = df_result.reindex(columns=all_cols)
    else:
        # all inputs were empty/all-NA
        df_result = pd.DataFrame().reindex(columns=all_cols)

    return df_result


def load_all_instances(
    catalog_root_directory: Path,
    catalog_filename: str,
    clouds: List[Cloud],
    gpus_only: bool,
    name_filter: Optional[str],
    region_filter: Optional[str],
    quantity_filter: Optional[int],
    case_sensitive: bool,
    all_regions: bool,
    drop_instance_type_none: bool,
) -> Dict[str, List[InstanceTypeInfo]]:
    dfs: List[pd.DataFrame] = []
    for cloud in clouds:
        catalog_file_path = catalog_root_directory / cloud.value / catalog_filename
        df_cloud: Optional[pd.DataFrame] = _load_instances_of_cloud(
            catalog_file_path,
            gpus_only,
            name_filter,
            region_filter,
            quantity_filter,
            case_sensitive,
            drop_instance_type_none,
        )
        if df_cloud is not None and not df_cloud.empty:
            df_cloud["Cloud"] = cloud.value
            dfs.append(df_cloud)

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

        rows = rows.sort_values(by=sort_key)
        rows = rows.astype(object).where(pd.notna(rows), None)
        ret = rows.apply(
            lambda row: InstanceTypeInfo(
                cloud=row["Cloud"],
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

    df_result: pd.DataFrame = _clean_concat_dataframes(dfs)
    if df_result.empty:
        return {}

    grouped = df_result.groupby("AcceleratorName")
    ret_dict: Dict[str, List[InstanceTypeInfo]] = {
        str(k): make_list_from_df(v) for k, v in grouped
    }
    return ret_dict


async def list_instances_stream(
    catalog_root_directory: Path,
    catalog_filename: str,
    clouds: List[Cloud],
    gpus_only: bool,
    name_filter: Optional[str],
    region_filter: Optional[str],
    quantity_filter: Optional[int],
    case_sensitive: bool,
    all_regions: bool,
    drop_instance_type_none: bool,
) -> AsyncGenerator[str, None]:
    results: Dict[str, List[InstanceTypeInfo]] = load_all_instances(
        catalog_root_directory,
        catalog_filename,
        clouds,
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
