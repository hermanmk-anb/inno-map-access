import pathlib
import re
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from tqdm import tqdm
from unidecode import unidecode

# Configuratie
PRODUCT = 1197
MAP_LAYERS = ["RGB", "CIR"]  # Kleur (RGB) en Infrarood (Vegetatie)
N_TILES = -1
DATA_PATH = pathlib.Path("/home/azureuser/cloudfiles/code/data/geo_images/")  # Pad naar de map waar de bestanden worden opgeslagen
N_WORKERS = 8

# Derivatives
URL_PRUDUCT = f"https://download.vlaanderen.be/bff/v1/Products/{PRODUCT}"
URL_ORDER = "https://download.vlaanderen.be/bff/v1/Orders"

HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def create_order(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Creates an order to download the files of the product based on the metadata and the configuration (MAP_LAYERS and N).

    :param metadata: info about the product and its files
    :return: the response of the order request, which contains the order id and the files that are being processed
    """
    # Create the order
    field_ids = []
    candidate_file_ids = {f"{i:02d}" for i in range(len(metadata["files"])) if N_TILES < 1 or i <= N_TILES}
    for file in metadata["files"]:
        # check if we want to download this file based on the layer (RGB or CIR)
        if not any(layer in file["fileId"] for layer in MAP_LAYERS):
            continue

        # check if we want to download this file based on the index (N)
        if re.search("([0-9]+).zip", file["fileId"]).group(1) not in candidate_file_ids:  # type: ignore
            continue

        # collect the filed ids we want to download
        field_ids.append(file["fileId"])

    # send the order request to download the files
    return requests.post(
        URL_ORDER,
        headers=HEADER,
        json={
            "productId": PRODUCT,
            "discreteVersionIds": [],
            "fileIds": field_ids,
            "geographicalCrop": {"clip": False},
            "mailOnProcessed": False,
        },
    ).json()


# Get the download urls
def get_download_urls(order: dict[str, Any]) -> Generator[dict[str, str]]:
    """
    Gets the download urls for the files in the order.

    :param order: _description_
    :yield: return a dictionary with the order id, file id, file name, folder name and the download url
    """
    for o in order["member"]:
        # get the order_id
        order_id = o["orderId"]

        # get the file metadata to find the download link
        file_metadata = requests.get(f"https://download.vlaanderen.be/bff/v1/Orders/{order_id}", headers=HEADER).json()

        # get the download url for the file
        download_file_id = file_metadata["downloads"][0]["fileId"]
        download_name = file_metadata["downloads"][0]["name"]

        # url
        yield {
            "order_id": order_id,
            "file_id": download_file_id,
            "file_name": download_name,
            "folder_name": download_name.rsplit("_", 1)[0],
            "url": f"https://download.vlaanderen.be/bff/v1/Orders/{order_id}/download/{download_file_id}",
        }


def download_file(download_info: dict[str, str], path: pathlib.Path) -> pathlib.Path:
    """Download a single file with a progress bar and save it in the correct folder."""
    # create the folder if it doesn't exist and get the file path
    folder = path / download_info["folder_name"]
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / download_info["file_name"]

    with requests.get(download_info["url"], headers=HEADER, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=download_info["file_name"]) as pbar:
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    return file_path


def main() -> None:

    # Get metadata about the product and its files
    metadata = requests.get(URL_PRUDUCT, headers=HEADER).json()

    # Create the order
    orders = create_order(metadata)

    # Get the download urls
    urls_2_download = list(get_download_urls(orders))

    # Get the root path to save the files
    product_name = unidecode(metadata["name"].lower())
    product_name = re.sub("[^a-zA-Z0-9]+", "_", product_name).strip("_")
    download_path = DATA_PATH / product_name

    # Download all files in parallel
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(download_file, info, download_path) for info in urls_2_download]
        for future in as_completed(futures):
            path = future.result()
            print(f"✅ Downloaded {path}")


if __name__ == "__main__":
    main()