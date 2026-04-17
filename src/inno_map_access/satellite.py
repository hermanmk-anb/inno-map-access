import logging
import sys
from datetime import date, datetime, timedelta

import dask.array as da
import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import shapely
import xarray
import zarr
from cachetools.func import lru_cache
from odc.stac import load
from omnicloudmask import predict_from_array
from scipy.signal import fftconvolve
from xarray.backends.common import T_PathFileOrDataStore
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.errors import GroupNotFoundError

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# minimal clear patch size (in pixels) before we consider an image patch worth saving.
CONTIGUITY_SIZES = [32, 64, 128, 256, 512, 1024]

COVER_MAP = {
    0: "invalid",
    1: "thick cloud",
    2: "thin cloud",
    3: "cloud shadow",
    4: "clear",
}

COVER_MAP.update({i + len(COVER_MAP): f"clear for {s} square pixels" for i, s in enumerate(CONTIGUITY_SIZES)})

CHUNK_SIZE = 512
SHARD_SIZE = 4096

COLOR_MAP = {
    0: "black",
    1: "white",
    2: "lightgrey",
    3: "dimgrey",
    4: "brown",
    5: "orange",
    6: "gold",
    7: "yellow",
    8: "greenyellow",
    9: "lime",
    10: "green",
}


def plot_cloud_cover_map(cc_array):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["figure.dpi"] = 400

    bounds = np.arange(len(COLOR_MAP))
    cmap = mpl.colors.ListedColormap([COLOR_MAP[i] for i in bounds])
    norm = mpl.colors.BoundaryNorm(list(bounds) + [11], cmap.N)

    fig, ax = plt.subplots()

    cax = ax.imshow(cc_array + 0.5, cmap=cmap, norm=norm)
    cbar = fig.colorbar(cax, orientation="horizontal")
    cbar.ax.set_xticks(bounds + 0.5)
    cbar.ax.set_xticklabels([COVER_MAP[i] for i in bounds], rotation=45, ha="right")

    plt.show()


@lru_cache
def get_flanders_geometry(epsg: int) -> shapely.MultiPolygon:
    return (
        gpd.read_file(
            "/home/azureuser/cloudfiles/code/data/vector_data/gewest_vlaanderen_brussel_2025/Refgew25G10_buff500.shp"
        )
        .to_crs(epsg=epsg)
        .iloc[0]
        .geometry
    )


def valid_mask(scl: np.ndarray | xarray.DataArray) -> np.ndarray:
    return np.isin(scl, [4, 5, 6, 11])


def contiguity_filter(
    mask: np.ndarray, n_contiguous_pixels: int, tolerance: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """
    For given pixel mask, select parts that have unmasked pixels within a square of size n_contiguous_pixels.

    This function returns two new masks:
    - The set of pixels that are upper left corners of a clear square of given size surrounding them
    - The set of pixels that are contained within an existing clear square of given size.

    The point of this function is to eliminate regions that contain a mix of masked and unmasked pixels.
    """
    corner_mask = open_square_corner(mask, n_contiguous_pixels, tolerance)
    contiguous_field = np.ones((n_contiguous_pixels, n_contiguous_pixels))
    expanded_mask = np.round(fftconvolve(corner_mask, contiguous_field, "full")) > 0
    return corner_mask, expanded_mask


def open_square_corner(mask: np.ndarray, n_contiguous_pixels: int, tolerance: float = 0.01) -> np.ndarray:
    if min(mask.shape) < n_contiguous_pixels:
        # if the mask does not fit the number of expected contiguous pixels at all, the convolution operator would
        # raise an error. We return an empty array
        return np.zeros((0, 0)).astype(bool)
    contiguous_field = np.ones((n_contiguous_pixels, n_contiguous_pixels))
    return np.round(fftconvolve(mask, contiguous_field, "valid")) >= (n_contiguous_pixels**2) * (1 - tolerance)


class EmptyCanvasError(RuntimeError):
    """Raised when no valid pixels are present"""


def compute_valid_box(truth_array: np.ndarray):
    x_bounds = truth_array.any(axis=0)
    if not x_bounds.any():
        raise EmptyCanvasError("truth array contains no valid pixels")
    y_bounds = truth_array.any(axis=1)
    xmin, xmax = np.argmax(x_bounds), (x_bounds.shape[0] - np.argmax(x_bounds[::-1]))
    ymin, ymax = np.argmax(y_bounds), (y_bounds.shape[0] - np.argmax(y_bounds[::-1]))
    if (ymax - ymin < CONTIGUITY_SIZES[0]) or (xmax - xmin < CONTIGUITY_SIZES[0]):
        raise EmptyCanvasError("truth array contains insufficient valid pixels")
    return ymin, ymax, xmin, xmax


def compute_bands(arr: da.Array, valid_pixels: np.ndarray) -> da.Array:
    arr = arr.rechunk(list(arr.shape[:-2]) + [None, None])
    vertical, horizontal = [[0] + list(np.cumsum(ch)) for ch in arr.chunks[-2:]]
    result = da.empty_like(arr)
    for u, d in zip(vertical[:-1], vertical[1:]):
        for l, r in zip(horizontal[:-1], horizontal[1:]):
            if valid_pixels[u:d, l:r].any():
                result[:, :, u:d, l:r] = arr[:, :, u:d, l:r].compute()
    return result


def get_lazy_array(date: date, max_cloud_cover: int = 50, resolution: int = 10) -> xarray.DataArray | None:
    flanders_4326 = get_flanders_geometry(4326)

    dt = datetime(date.year, date.month, date.day)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=flanders_4326,
        datetime=(dt, dt + timedelta(1)),
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    ).item_collection()

    if search:
        bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL"]

        return load(
            items=search,
            intersects=flanders_4326,
            resolution=resolution,
            bands=bands,
            dtype="uint16",
            chunks={"x": CHUNK_SIZE, "y": CHUNK_SIZE, "time": 1},
            groupby="solar_day",
        ).to_array(dim="band", name="sentinel-2-l2a")


def process_day(
    day: date, max_cloud_cover: int = 50, resolution: int = 10
) -> tuple[xarray.DataArray, xarray.DataArray, np.ndarray] | None:

    result = get_lazy_array(day, max_cloud_cover, resolution)
    if result is not None:
        logging.info(f"Found images for {day}")
        flanders_geom = get_flanders_geometry(int(result.coords["spatial_ref"]))
        X, Y = np.meshgrid(result.coords["x"].data, result.coords["y"].data)
        in_flanders = shapely.contains_xy(flanders_geom, X, Y)
        valid_pixels = (result.data[[12]].compute() != 0)[0, 0] & in_flanders
        try:
            ymin, ymax, xmin, xmax = compute_valid_box(valid_pixels)

            red_green_nir = result.data[[3, 2, 7], 0, ymin:ymax, xmin:xmax].compute()

            logger.info(" - predicting cloud coverage...")
            cloud_cover = predict_from_array(red_green_nir)

            logger.info(" - creating patch coverage map")

            valid_and_cloud_free = valid_pixels[ymin:ymax, xmin:xmax] & (cloud_cover[0] == 0)
            contiguous_cover_map = np.zeros_like(valid_and_cloud_free).astype("uint8")
            logger.info(" - creating coverage map...")
            for patch_size in CONTIGUITY_SIZES:
                c_cloud_cover = open_square_corner(valid_and_cloud_free, patch_size, tolerance=0)
                contiguous_cover_map[: c_cloud_cover.shape[0], : c_cloud_cover.shape[1]] += c_cloud_cover
            coverage_map = np.zeros_like(valid_pixels, dtype="uint8")
            coverage_map[ymin:ymax, xmin:xmax] = (4 - cloud_cover) * valid_pixels[ymin:ymax, xmin:xmax]
            coverage_map[ymin:ymax, xmin:xmax] += contiguous_cover_map.astype("uint8")

            rc = result.coords
            cloud_cover = xarray.DataArray(
                name="cloud_cover",
                data=coverage_map[None, :, :],
                coords=[rc["time"], rc["y"], rc["x"]],
            )
            cloud_cover = cloud_cover.assign_coords(spatial_ref=result.coords["spatial_ref"])
            cloud_cover.data = da.from_array(cloud_cover.data, chunks=(1, CHUNK_SIZE, CHUNK_SIZE))

            logger.info(f" - computing contiguity mask of {CONTIGUITY_SIZES[0]} pixels...")
            _, e_cloud_cover = contiguity_filter(valid_and_cloud_free, CONTIGUITY_SIZES[0], tolerance=0.05)
            valid_pixels[ymin:ymax, xmin:xmax] = e_cloud_cover

            logger.info(" - downloading valid patches...")

            # Recreate lazy array, as links for a pystac item search only remain valid for a short period.
            result = get_lazy_array(day, max_cloud_cover, resolution)
            result.data = compute_bands(result.data, valid_pixels)

            return result, cloud_cover, valid_pixels

        except EmptyCanvasError:
            logging.info(f"no valid pixels found for {day}. Skipping")
    else:
        logger.info(f"No items found for {day}")


def append_to_zarr(store: T_PathFileOrDataStore, arr: xarray.DataArray):
    arr = arr.transpose("time", "y", "x", ...)
    chunk_sizes = [ch[0] for ch in arr.chunks]
    arr.coords["time"] = arr.coords["time"].astype("int64")

    shard_sizes = chunk_sizes.copy()
    shard_sizes[:3] = [1, SHARD_SIZE, SHARD_SIZE]
    arr.data.rechunk(shard_sizes)
    exists = True
    try:
        xarray.open_dataarray(store, engine="zarr")
        logging.info(" - appending to existing zarr store...")
    except (GroupNotFoundError, FileNotFoundError):
        logging.info(" - no zarr array found. Will create from scratch ...")
        exists = False
    arr.to_zarr(
        store=store,
        mode="a-" if exists else "w",
        append_dim="time" if exists else None,
        encoding=None
        if exists
        else {
            arr.name: {
                "chunks": chunk_sizes,
                "shards": shard_sizes,
                "compressors": BloscCodec(cname="lz4", clevel=2, shuffle=BloscShuffle.shuffle),
                "write_empty_chunks": False,
                "_FillValue": 0,
            }
        },
        compute=False,
        safe_chunks=not exists,
    ).compute()


if __name__ == "__main__":
    sentinel_store = "/home/azureuser/cloudfiles/code/data/sattelite/sentinel_2_sample"
    sentinel_cloud_cover_store = "/home/azureuser/cloudfiles/code/data/sattelite/cloud_cover"

    try:
        handle = xarray.open_dataarray(sentinel_store, engine="zarr")
        last_date = datetime.fromtimestamp(handle.coords["time"].data[-1] / 10**6).date()

    except (GroupNotFoundError, FileNotFoundError):
        last_date = date(2015, 8, 30)

    today = date.today()
    day_count = (today - last_date).days
    for day in [d for d in (last_date + timedelta(n + 1) for n in range(day_count))]:
        r = process_day(day, resolution=100)
        if r is not None:
            result, patch_coverage, valid_pixels = r
            append_to_zarr(sentinel_store, result)
            append_to_zarr(sentinel_cloud_cover_store, patch_coverage)
