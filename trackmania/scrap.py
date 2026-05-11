import os
from concurrent.futures import ThreadPoolExecutor

import requests
from loguru import logger

API_URL = "https://api.polyhaven.com"
OUTPUT_DIR = "polyhaven_textures"
TEXTURES_FORMAT = ["Diffuse", "nor_gl", "Rough"]
IMG_FORMAT = "png"
IMG_RES = "1k"

MAX_WORKERS = 8

def get_all_textures():
    """
    Fetch all texture assets from Poly Haven API
    """
    url = f"{API_URL}/assets?t=textures"
    
    response = requests.get(url)
    return response.json()

def get_texture_files(texture_id):
    """
    Fetch downloadable files for a texture.
    """
    url = f"{API_URL}/files/{texture_id}"

    response = requests.get(url)
    return response.json()

def download_file(path, url):
    """
    Download a single file if it doesn't already exist
    """
    if os.path.exists(path):
        logger.info(f"[SKIP] {path}")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger.info(f"[DOWNLOADING] {path}")
    with requests.get(url, stream=True) as r:
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    logger.info(f"[DONE] {path}")

def collect_downloads(file_data):
    """
    Extractr all downloable texture maps.
    """
    downloads = []

    # Example structure:
    # file_data["Diffuse"]["1k"]["png"]["url"]

    for format in TEXTURES_FORMAT:
        if format not in file_data.keys():
            continue
        url = file_data[format][IMG_RES][IMG_FORMAT]['url']
        filename = os.path.basename(url)
        downloads.append((os.path.join(OUTPUT_DIR, filename), url))

    return downloads


def main():
    textures = get_all_textures()
    
    logger.info(f"Found {len(textures)} textures")
    downloads = []

    for texture_id in textures.keys():
        try:
            textures = get_texture_files(texture_id=texture_id)
            downloads.extend(collect_downloads(file_data=textures))
            logger.info(f"\nFetching files for: {texture_id}")
        except Exception as e:
            logger.error(f"[ERROR] {texture_id}: {e}")

    logger.info(f"\nTotal files to process: {len(downloads)}")
 

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for path, url in downloads:
            futures.append(executor.submit(download_file, path, url))

        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.info(f"[DOWNLOAD ERROR] {e}")


if __name__ == "__main__":
    main() 