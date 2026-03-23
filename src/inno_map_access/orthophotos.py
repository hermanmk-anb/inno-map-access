import os
from datetime import date
from pathlib import Path

import numpy as np
import rasterio


class GeoRasterBase:
    @classmethod
    def folder(cls) -> str:
        raise NotImplementedError

    @classmethod
    def file_extension(cls) -> str:
        raise NotImplementedError

    def __init__(self) -> None:
        self.files = [str(p.absolute()) for p in Path(self.folder()).iterdir()]
        self.files = [f for f in self.files if f.endswith(self.file_extension()) and os.path.isfile(f)]

    def get_subimage_on_pixels(
        self, top: float, left: float, n_pixels_vertical: int, n_pixels_horizontal: int
    ) -> np.ndarray:
        for file in self.files:
            with rasterio.open(file):
                pass

    def get_subimage_on_size(
        self, top: float, left: float, height_in_meters: float, width_in_meters: float
    ) -> np.ndarray:
        pass


class OrthophotoBase(GeoRasterBase):
    @classmethod
    def date_polygon_file(cls) -> str:
        raise NotImplementedError

    def get_location_image_date(self, lat: float, lon: float) -> date:
        pass
