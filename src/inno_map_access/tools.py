import numpy as np
import rasterio
from matplotlib.pyplot import imshow
from rasterio.io import DatasetReader
from rasterio.warp import transform
from rasterio.windows import from_bounds


def sample_patch_from_image(
    image_src: DatasetReader,
    lon_left: float,
    lat_up: float,
    lon_right: float,
    lat_down: float,
) -> np.ndarray:
    v_coords, h_coords = transform(
        "EPSG:4326", image_src.crs, [lon_left, lon_right], [lat_up, lat_down]
    )

    image_src.index(v_coords[0], h_coords[0])
    image_src.index(v_coords[1], h_coords[1])

    return image_src.read(
        window=from_bounds(
            v_coords[0], h_coords[1], v_coords[1], h_coords[0], image_src.transform
        )
    )


photo_2015 = rasterio.open(
    "/home/azureuser/cloudfiles/code/data/geo_images/orthofotomozaiek_middenschalig_winteropnamen_kleur_2015_vlaanderen/OMWRGB15VL/JPEG2000/combined.vrt"
)
crown_cover = rasterio.open(
    "/home/azureuser/cloudfiles/code/data/geo_images/kruinbedekking/BGK2015_1/GeoTIFF/combined.vrt"
)

photo = sample_patch_from_image(photo_2015, 4.727940, 50.888077, 4.736795, 50.884715)
cover = sample_patch_from_image(crown_cover, 4.727940, 50.888077, 4.736795, 50.884715)

imshow(photo.transpose(1, 2, 0))
imshow(cover.transpose(1, 2, 0))
