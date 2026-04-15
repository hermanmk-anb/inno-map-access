import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform
from rasterio.windows import from_bounds
from scipy.sparse import csr_array
from shapely.affinity import affine_transform
from shapely.geometry.base import BaseGeometry

# LAT LON BOUNDS
LEFT, TOP, RIGHT, BOTTOM = 4.727940, 50.888077, 4.736795, 50.884715


class ReSampler:
    def __init__(self, n_input: int, n_target: int) -> None:
        i_range = np.arange(n_target)[:, None]
        j_range = np.arange(n_input)[None, :]
        upper = (j_range + 1) * n_target / n_input
        lower = j_range * n_target / n_input
        self.T = np.maximum(i_range, np.minimum(i_range + 1, upper)) - np.maximum(
            i_range, np.minimum(i_range + 1, lower)
        )
        self.T /= np.sum(self.T, axis=1, keepdims=True)
        self.T_sparse = csr_array(self.T)

    def resample_sparse(self, x: np.ndarray) -> np.ndarray:
        return np.vstack([(self.T_sparse @ x[i])[None, ...] for i in np.arange(x.shape[0])])

    def resample(self, x: np.ndarray) -> np.ndarray:
        return self.T @ x


class ReGridder:
    """
    I regrid nothing! A compact class to be used for (repeated) simple resizing of a picture from one number of pixels
    to another.
    """

    def __init__(self, input_shape: tuple[int, int], target_shape: tuple[int, int]):
        self.vertical_resampler = ReSampler(input_shape[0], target_shape[0])
        self.horizontal_resampler = ReSampler(input_shape[1], target_shape[1])

    def regrid(self, image_array: np.ndarray, astype: np.dtype | None = None) -> np.ndarray:
        v_im = self.vertical_resampler.resample_sparse(image_array)
        h_im = self.horizontal_resampler.resample_sparse(v_im.transpose(0, 2, 1))
        result = h_im.transpose(0, 2, 1)
        dtype = astype or image_array.dtype
        return result.astype(dtype)


def geopandas_transform(transform: Affine, geometry: BaseGeometry) -> BaseGeometry:
    return affine_transform(
        geometry,
        [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f],
    )


# Create image handles
photo_2015 = rasterio.open(
    "/home/azureuser/cloudfiles/code/data/geo_images/orthofotomozaiek_middenschalig_winteropnamen_kleur_2015_vlaanderen/OMWRGB15VL/JPEG2000/combined.vrt"
)
crown_cover_grid = rasterio.open(
    "/home/azureuser/cloudfiles/code/data/geo_images/kruinbedekking/BGK2015_1/GeoTIFF/combined.vrt"
)

# Coordinates for image patch, in longitudes and latitudes
v_coords, h_coords = transform("EPSG:4326", photo_2015.crs, [LEFT, RIGHT], [TOP, BOTTOM])

bbox = v_coords[0], h_coords[1], v_coords[1], h_coords[0]
photo_window = from_bounds(*bbox, transform=photo_2015.transform)
window_transform = photo_2015.window_transform(photo_window)
coords_to_pixels = ~window_transform
# sample the patches from the complete image
photo = photo_2015.read(window=photo_window)
cover = crown_cover_grid.read(window=from_bounds(*bbox, transform=crown_cover_grid.transform))

cover_resampled = ReGridder(cover.shape[1:], photo.shape[1:]).regrid(cover)

# point markings
trunks = gpd.read_file(
    "/home/azureuser/cloudfiles/code/data/geo_images/Gedetailleerd_Groen_Vlaanderen_LiDAR_2015_entiteiten/lc_gg_stam_2015.gpkg",
    bbox=bbox,
)
crown_tips = gpd.read_file(
    "/home/azureuser/cloudfiles/code/data/geo_images/Gedetailleerd_Groen_Vlaanderen_LiDAR_2015_entiteiten/lc_gg_kruintop_2015.gpkg",
    bbox=bbox,
)


trunk_positions = coords_to_pixels * (trunks.geometry.x, trunks.geometry.y)
crown_tip_positions = coords_to_pixels * (crown_tips.geometry.x, crown_tips.geometry.y)

# polygons:
bushes = gpd.read_file(
    "/home/azureuser/cloudfiles/code/data/geo_images/Gedetailleerd_Groen_Vlaanderen_LiDAR_2015_entiteiten/lc_gg_groen_300cm_2015.gpkg",
    bbox=bbox,
)

crowns = gpd.read_file(
    "/home/azureuser/cloudfiles/code/data/geo_images/Gedetailleerd_Groen_Vlaanderen_LiDAR_2015_entiteiten/lc_gg_kruin_2015.gpkg",
    bbox=bbox,
)

bushes["geometry"] = bushes["geometry"].apply(lambda x: geopandas_transform(~window_transform, x))
crowns["geometry"] = crowns["geometry"].apply(lambda x: geopandas_transform(~window_transform, x))


# Show some plots
import matplotlib
import matplotlib.pyplot as plt

# set resolution to a useable level
matplotlib.rcParams["figure.dpi"] = 600

_, ax = plt.subplots()

# # show photo patch
# plt.imshow(photo.transpose(1, 2, 0))
# plt.show()
# # show crown cover patch
# plt.imshow(cover.transpose(1, 2, 0))
# plt.show()

# show overlay, where the red channel has been replaced with the resampled crown cover patch
# ax.imshow(
#     np.vstack(
#         [
#             127 * cover_resampled,
#             photo[1:3],
#         ]
#     ).transpose(1, 2, 0)
# )
ax.imshow(photo.transpose(1, 2, 0))
ax.scatter(*trunk_positions, s=2, c="b", marker="x", linewidths=0.25)
ax.scatter(*crown_tip_positions, s=2, c="g", marker="+", linewidths=0.25)
bushes.boundary.plot(ax=ax, linewidth=0.25, edgecolor="y")
crowns.boundary.plot(ax=ax, linewidth=0.25, edgecolor="g")
plt.show()
