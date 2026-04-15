import time

import numpy as np
import rioxarray
import xarray as xr
import zarr
from affine import Affine
from obstore.auth.azure import AzureCredentialProvider
from obstore.store import AzureStore
from pyproj import CRS
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import ObjectStore

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
rioxarray.show_versions()
# 1. Configuration & Authentication
# Fabric OneLake URI: https://onelake.dfs.fabric.microsoft.com/<workspace>/<item>.<type>/<path>
WORKSPACE_ID = "23b1515f-7adb-4ee2-bebe-8e41b9f5f2d2"
LAKEHOUSE_ID = "cbb46753-ae96-4613-a29f-b09eeeeb1746"
FABRIC_GEO_IMAGE = "data_cube"


def get_onelake_zarr_store(
    workspace_id: str,
    lakehouse_id: str,
    fabric_geo_image_name: str,
    read_only: bool = True,
) -> ObjectStore:
    """
    Create a Zarr-compatible ObjectStore backed by Azure OneLake storage.

    This function configures an AzureStore pointing to a specific path inside
    a Microsoft Fabric Lakehouse and wraps it in a Zarr ObjectStore for use
    with Zarr-based data access.

    Parameters
    ----------
    workspace_id : str
        The Microsoft Fabric workspace ID (used as the container name).
    lakehouse_id : str
        The Lakehouse ID within the workspace.
    fabric_geo_image_name : str
        The name of the dataset or folder under the Lakehouse "Files" directory.
    read_only : bool, optional
        If True, the returned store will be read-only. Default is True.

    Returns
    -------
    ObjectStore
        A Zarr-compatible object store backed by Azure OneLake.
    """

    # Initialize Azure credential provider (handles authentication)
    credential_provider = AzureCredentialProvider()

    azure_store = AzureStore(
        account_name="onelake",  # OneLake storage account
        container_name=workspace_id,  # Workspace acts as container
        prefix=f"{lakehouse_id}/Files",  # Path to the data
        use_fabric_endpoint=True,  # Use Microsoft Fabric endpoint
        credential_provider=credential_provider,  # Authentication handler
    )
    # Wrap AzureStore in a Zarr ObjectStore interface
    return ObjectStore(azure_store, read_only=read_only)


ONELAKE_STORE = get_onelake_zarr_store(
    workspace_id=WORKSPACE_ID,
    lakehouse_id=LAKEHOUSE_ID,
    fabric_geo_image_name=FABRIC_GEO_IMAGE,
    read_only=False,
)


def create_geozarr_metadata(
    ds: xr.Dataset,
    crs: str = "EPSG:31370",
    x_res: float = 0.4,
    y_res: float = -0.4,
    origin_x: float = 129999.8,
    origin_y: float = 249999.8,
) -> xr.Dataset:

    transform = Affine.translation(origin_x, origin_y) * Affine.scale(x_res, -y_res)

    ds = ds.rio.write_crs(crs)
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds = ds.rio.write_transform(transform)

    cf_attrs = CRS.from_user_input(crs).to_cf()

    ds["spatial_ref"] = xr.DataArray(0, attrs=cf_attrs)

    for var in ds.data_vars:
        ds[var].attrs["grid_mapping"] = "spatial_ref"

    return ds


def main():

    # --------------------------------------------------
    # Create a Zarr root group
    # --------------------------------------------------
    root = zarr.open_group(
        store=ONELAKE_STORE,
        zarr_format=3,
    )

    # --------------------------------------------------
    # Shape
    # --------------------------------------------------
    canvas_shape = (1024, 1024)
    shard_shape = (256, 256)
    chunk_shape = (32, 32)

    # coordinates
    crs = "EPSG:31370"
    x_res, y_res = 0.4, -0.4
    origin_x, origin_y = 129999.8, 249999.8

    # Generate coordinate arrays (in Lambert 72 meters)
    x_coords = origin_x + np.arange(canvas_shape[1]) * x_res
    y_coords = origin_y + np.arange(canvas_shape[0]) * y_res

    datums = np.array(["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"], dtype="datetime64[ns]")

    # --------------------------------------------------
    # Dummy L1 data
    # --------------------------------------------------
    # L1: sensor data
    # L1 sensor bands: R,G,B,NIR
    l1_data = np.zeros((len(datums), *canvas_shape, 4), dtype="uint8")
    l1_data[..., :] = [80, 160, 200, 240]  # base values

    # vary L1 values per month
    l1_data[0, :, :, :] //= 8
    l1_data[1, :, :, :] //= 4
    l1_data[2, :, :, :] //= 2
    # last month stays full

    #
    # create L1 array
    #
    ds_l1 = xr.Dataset(
        {"AERIAL_PHOTOGRAPHS": (["time", "y", "x", "band"], l1_data)},
        coords={
            "time": datums,
            "y": y_coords,
            "x": x_coords,
            "band": np.array(["R", "G", "B", "NIR"], dtype="object"),
        },
        attrs={"description": "Level 1 - Aerial Photographs- Raw sensor bands"},
    )
    ds_l1 = create_geozarr_metadata(ds_l1, crs, x_res, y_res, origin_x, origin_y)

    # Write L1 with your preferred chunking/sharding
    ds_l1.to_zarr(
        store=ONELAKE_STORE,
        group="L1",
        mode="w",
        consolidated=False,
        encoding={
            "AERIAL_PHOTOGRAPHS": {
                "shards": (1, *shard_shape, 4),
                "chunks": (1, *chunk_shape, 4),
                "compressors": BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),
                "write_empty_chunks": False,
            }
        },
    )

    # --------------------------------------------------
    # L2: derived indices
    # --------------------------------------------------
    l1 = root["L1"]["AERIAL_PHOTOGRAPHS"][:]
    r = l1[..., 0].astype(np.float32)
    nir = l1[..., 3].astype(np.float32)

    # compute NDVI and Moisture Index
    ndvi = (nir - r) / (nir + r)  # type:ignore
    mi = nir / (nir + r)  # type:ignore

    #
    # L2 NDVI & MI
    #
    ds_l2 = xr.Dataset(
        {
            "NDVI": (["time", "y", "x"], ndvi.data),
            "MI": (["time", "y", "x"], mi.data),
        },
        coords={
            "time": datums,
            "y": y_coords,
            "x": x_coords,
        },
        attrs={"description": "Level 2 - Vegetation indices"},
    )
    ds_l2 = create_geozarr_metadata(ds_l2, crs, x_res, y_res, origin_x, origin_y)

    ds_l2.to_zarr(
        store=ONELAKE_STORE,
        group="L2",
        mode="w",
        consolidated=False,
        encoding={
            "NDVI": {
                "shards": (1, *shard_shape),
                "chunks": (1, *chunk_shape),
                "compressors": BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),
                "write_empty_chunks": False,
            },
            "MI": {
                "shards": (1, *shard_shape),
                "chunks": (1, *chunk_shape),
                "compressors": BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),
                "write_empty_chunks": False,
            },
        },
    )

    # --------------------------------------------------
    # L3: aggregated NDVI (monthly average)
    # --------------------------------------------------
    l3_ndvi = ndvi.mean(axis=0)

    #
    # L3
    #
    ds_l3 = xr.Dataset(
        {"monthly_avg_NDVI": (["y", "x"], l3_ndvi.data)},
        coords={"y": y_coords, "x": x_coords},
        attrs={"description": "Level 3 - Monthly average NDVI"},
    )
    ds_l3 = create_geozarr_metadata(ds_l3, crs, x_res, y_res, origin_x, origin_y)

    ds_l3.to_zarr(
        store=ONELAKE_STORE,
        group="L3",
        mode="w",
        consolidated=False,
        encoding={
            "monthly_avg_NDVI": {
                "shards": shard_shape,
                "chunks": chunk_shape,
                "compressors": BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),
                "write_empty_chunks": False,
            }
        },
    )

    # -------------------------------------------------------
    # Read data
    # ------------------------------------------------------

    #
    # Zarr, pixels & indices
    #
    # open a z-array
    z_arr = zarr.open(store=ONELAKE_STORE, path="L1/AERIAL_PHOTOGRAPHS", mode="r")

    # display some chunks
    print(z_arr[0, :5, :5, 3])

    #
    # xarray
    #

    # 1. Open the L1 dataset from the store
    ds_read = xr.open_zarr(store=ONELAKE_STORE, group="L1", consolidated=False, decode_coords="all")
    print(ds_read)
    # 2. Select data
    target_dates = ["2026-02-01", "2026-03-01"]

    # BBox: [minx, miny, maxx, maxy]
    bbox = [129999, 249950, 130100, 249999]

    # 3. Perform the selection
    # .sel(time=...) filters the dates
    # .sel(band="NIR") gets the specific band
    # .rio.clip_box(...) handles the spatial slicing using Lambert 72
    print(ds_read["AERIAL_PHOTOGRAPHS"])
    subset = (
        ds_read["AERIAL_PHOTOGRAPHS"]
        .sel(time=target_dates, band="NIR")
        .rio.clip_box(
            minx=bbox[0],
            miny=bbox[1],
            maxx=bbox[2],
            maxy=bbox[3],
        )
    )

    print("--- Selection Results ---")
    print(subset)  # LAzy
    print(subset.values)  # To bring it into memory as a numpy array:

    # #
    # # rasterio
    # # - Rasterio is niet gemaakt voor meerdere dimensies ... :(
    # path = get_rasterio_store(ONELAKE_STORE, "L1/AERIAL_PHOTOGRAPHS/0")
    # print(path)

    # with rasterio.open(path) as src:
    #     pass

    #     # 2. Define your query parameters
    #     # Rasterio doesn't "know" band names like 'NIR' by default unless
    #     # stored in metadata. Usually, you use the index.
    #     # Let's assume NIR is index 4 and we want the first two time slices.
    #     band_index = 4
    #     bbox = [129999, 249950, 130100, 249999]  # [minx, miny, maxx, maxy]

    #     # 3. Calculate the 'Window'
    #     # This converts coordinate bounds into pixel offsets (row/col)
    #     window = from_bounds(*bbox, transform=src.transform)

    #     # 4. Perform the selection
    #     # We read specific indexes. Note: Rasterio is 1-indexed for bands.
    #     # If the Zarr is 3D (Time, Height, Width), we treat time slices
    #     # as indexes or separate subdatasets.
    #     subset = src.read(band_index, window=window)

    #     print("--- Selection Results ---")
    #     print(f"Shape: {subset.shape}")
    #     print(subset)


if __name__ == "__main__":
    print("Starting Training Loop...")
    start_time = time.time()
    main()
    print(f"Total time: {time.time() - start_time:.2f}s")
