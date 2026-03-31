import planetary_computer
import pystac_client

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

time_range = "2026-01-01/2026-01-10"
bbox = [2.418407, 50.705241, 6.001626, 51.496387]

search = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=time_range)
item = search.get_all_items()
