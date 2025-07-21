# import system modules
import sys
import os
import json

# import external modules
import pytesseract
from PIL import Image

""" Module documentation
"""


def extract_text_from_image(image_path):
    """ Extracts texts from machinery image. The texts returned are:
        Part Number / Serial Number / Aircraft ID / Engine Number / Time on Wing
    """
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return (text)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__=="__main__":
    images_dir = os.path.abspath('images')
    texts = {}
    for image in os.listdir('images'):
        image_path = images_dir + '/' + image
        texts[image] = extract_text_from_image(image_path)
    with open('output.json', 'w') as f:
        json.dump(texts, f, indent=3)