from inno_map_access.orthophotos import OrthophotoBase


class FlandersWinter2025(OrthophotoBase):
    @classmethod
    def folder(cls) -> str:
        return "/home/azureuser/localfiles/projects/ee_experiment/data/orthofotos/JPEG2000"

    @classmethod
    def file_extension(cls) -> str:
        return ".jp2"


class BelgianTreeCrowns2015(OrthophotoBase):
    @classmethod
    def folder(cls) -> str:
        return "/home/azureuser/localfiles/projects/ee_experiment/data/orthofotos/GeoTIFF"

    @classmethod
    def file_extension(cls) -> str:
        return "32.tif"
