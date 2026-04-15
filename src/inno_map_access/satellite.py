import logging
import sys
from datetime import datetime, timedelta

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
import shapely
import xarray
import zarr
from odc.stac import load
from omnicloudmask import predict_from_array
from scipy.signal import fftconvolve

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

# minimal clear patch size (in pixels) before we consider an image patch worth saving.
MIN_CONTIGUITY = 64
CHUNK_SIZE = 256
SHARD_SIZE = 2048


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

    return ymin, ymax, xmin, xmax


def process_day(date: datetime, max_cloud_cover: int = 50):
    flanders_4326 = get_flanders_geometry(4326)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=flanders_4326,
        datetime=(date, date + timedelta(1)),
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    ).item_collection()

    if search:
        logger.info(f"found {len(search)} items for {date} with less than {max_cloud_cover}% cloud cover.")
        bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL"]

        result = load(
            items=search,
            intersects=flanders_4326,
            resolution=10,
            bands=bands,
            dtype="uint16",
            chunks={"x": CHUNK_SIZE, "y": CHUNK_SIZE, "time": 1},
            groupby="solar_day",
        ).to_array(dim="band")
        logger.info(f"loaded virtual data cube for {date}")

        flanders_geom = get_flanders_geometry(int(result.coords["spatial_ref"]))
        X, Y = np.meshgrid(result.coords["x"].data, result.coords["y"].data)
        in_flanders = shapely.contains_xy(flanders_geom, X, Y)
        valid_pixels = (result.data[[12]].compute() != 0)[0, 0] & in_flanders
        try:
            ymin, ymax, xmin, xmax = compute_valid_box(valid_pixels)

            red_green_nir = result.data[[3, 2, 7], 0, ymin:ymax, xmin:xmax].compute()

            logger.info("predicting cloud coverage...")
            cloud_cover = predict_from_array(red_green_nir)

            logger.info("Creating patch coverage map")
            coverage_map = valid_pixels.astype("uint8")
            valid_and_cloud_free = coverage_map[ymin:ymax, xmin:xmax] * (cloud_cover[0] == 0)
            coverage_map[ymin:ymax, xmin:xmax] = valid_and_cloud_free
            for k, patch_size in enumerate([16, 32, 64, 128, 256, 512]):
                logger.info(f"creating layer for {patch_size} pixels ({k} out of 6 sizes)")
                c_cloud_cover = open_square_corner(valid_and_cloud_free, patch_size)
                coverage_map[ymin : ymin + c_cloud_cover.shape[0], xmin : xmin + c_cloud_cover.shape[1]][
                    c_cloud_cover
                ] = k + 2

            rc = result.coords
            patch_coverage = xarray.DataArray(
                data=coverage_map[None, :, :],
                coords=[rc["time"], rc["y"], rc["x"]],
            )
            patch_coverage = patch_coverage.assign_coords(spatial_ref=result.coords["spatial_ref"])

            logger.info(f"computing contiguity mask of {MIN_CONTIGUITY} pixels...")
            _, e_cloud_cover = contiguity_filter(valid_and_cloud_free, MIN_CONTIGUITY)
            valid_pixels[ymin:ymax, xmin:xmax] = e_cloud_cover
            return result, patch_coverage, valid_pixels

        except EmptyCanvasError:
            print(f"no valid pixels found for {date}. Skipping")
    else:
        print(f"No items found for {date}")


if __name__ == "__main__":
    result, patch_coverage, valid_pixels = process_day(datetime(2026, 4, 1))

    plt.imshow(patch_coverage.data[0, :, :])
    plt.colorbar()
