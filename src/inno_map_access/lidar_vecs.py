import geopandas as gpd



trunks = gpd.read_file(
    "/home/azureuser/cloudfiles/code/data/geo_images/Gedetailleerd_Groen_Vlaanderen_LiDAR_2015_entiteiten/lc_gg_stam_2015.gpkg"
)
bushes_pg = gpd.read_file(
    "/home/azureuser/cloudfiles/code/data/geo_images/Gedetailleerd_Groen_Vlaanderen_LiDAR_2015_entiteiten/lc_gg_groen_70cm_2015.gpkg"
)
crowntips = gpd.read_file(
    "/home/azureuser/cloudfiles/code/data/geo_images/Gedetailleerd_Groen_Vlaanderen_LiDAR_2015_entiteiten/lc_gg_kruintop_2015.gpkg"
)
crowns_pg = 