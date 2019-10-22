# code from https://github.com/BBarbosa/tflearn-image-recognition-toolkit/blob/4a0528dcfb206b1e45997f2fbc097aafacfa0fa0/scripts/html_link_parser.py

import re
import argparse

from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from skimage import io as skio
from urllib.request import urlopen
import os


def html_url_parser(url, save_dir, show=False, wait=False):
    """
    HTML parser to download images from URL.
    Params:\n
    `url` - Image url\n
    `save_dir` - Directory to save extracted images\n
    `show` - Show downloaded image\n
    `wait` - Press key to continue executing
    """

    website = urlopen(url)
    html = website.read()

    soup = BeautifulSoup(html, "html5lib")

    for image_id, link in enumerate(soup.find_all("a", href=True)):
        if image_id == 0:
            continue

        img_url = link["href"]

        try:
            if os.path.isfile(save_dir + "img-%d.png" % image_id) == False:
                print("[INFO] Downloading image from URL:", link["href"])
                image = Image.open(urlopen(img_url))
                image.save(save_dir + "img-%d.png" % image_id, "PNG")
                if show:
                    image.show()
            else:
                print("skipped")
        except KeyboardInterrupt:
            print("[EXCEPTION] Pressed 'Ctrl+C'")
            break
        except Exception as image_exception:
            print("[EXCEPTION]", image_exception)
            continue

        if wait:
            key = input("[INFO] Press any key to continue ('q' to exit)... ")
            if key.lower() == "q":
                break


# ///////////////////////////////////////////////////
#                   Main method
# ///////////////////////////////////////////////////
if __name__ == "__main__":
    URL_TRAIN_IMG = (
        "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html"
    )
    URL_TRAIN_GT = (
        "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html"
    )

    URL_TEST_IMG = (
        "https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/index.html"
    )
    URL_TEST_GT = (
        "https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/index.html"
    )

    html_url_parser(url=URL_TRAIN_IMG, save_dir="./road_segmentation/training/input/")
    html_url_parser(url=URL_TRAIN_GT, save_dir="./road_segmentation/training/output/")

    html_url_parser(url=URL_TEST_IMG, save_dir="./road_segmentation/testing/input/")
    html_url_parser(url=URL_TEST_GT, save_dir="./road_segmentation/testing/output/")

    print("[INFO] All done!")
