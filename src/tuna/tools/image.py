from pathlib import Path
from PIL import Image
import requests

def load(image_source):
    if image_source.startswith(("http://", "https://")):
        try:
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            return Image.open(response.raw)
        except Exception as ex:
            raise ValueError(f"{image_source=} failed to load: {ex}")
    elif Path(image_source).is_file():
        try:
            return Image.open(image_source)
        except IOError as ex:
            raise ValueError(f"{image_source=} failed to load: {ex}")
    else:
        raise ValueError(f"{image_source=} must be a valid URL or existing file.")
