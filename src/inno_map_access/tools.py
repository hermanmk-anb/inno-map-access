import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.warp import transform
from rasterio.windows import Window

from inno_map_access.scratch import FlandersWinter2025


def latlon_to_pixels(image_src: DatasetReader, lon: float, lat: float) -> tuple[float, float]:
    pxv, pxh = transform("EPSG:4326", image_src.crs, [lon], [lat])
    return tuple([np.floor(x) for x in src.index(pxv[0], pxh[0])])


def create_pixel_window_from_top_left_corner(
    image_src: DatasetReader,
    lat: float,
    lon: float,
    npx_vertical: int,
    npx_horizontal: int,
) -> Window:
    top, left = latlon_to_pixels(image_src, lat, lon)
    return Window(left, top, npx_horizontal, npx_vertical)


vld = FlandersWinter2025()
src = rasterio.open("/home/azureuser/localfiles/projects/ee_experiment/data/orthofotos/JPEG2000/combined.vrt")
image = src.read(window=create_pixel_window_from_top_left_corner(src, 5.605806, 50.991447, 1024, 2048))
