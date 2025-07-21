# import system modules
import sys
import os

# import external modules
import pytesseract
from PIL import Image

""" Module documentation
"""

# Optional: Set the tesseract command path (needed on Windows if not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text_from_image(image_path):
    """ Extracts texts from machinery image. The texts returned are:
        Part Number / Serial Number / Aircraft ID / Engine Number / Time on Wing
    """
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    try:
        image = Image.open(image_path)
        print("Image loaded successfully.")

        text = pytesseract.image_to_string(image)
        print("\n--- Extracted Text ---")
        print(text)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__=="__main__":
    images_dir = os.path.abspath('images')
    texts = {}
    for image in os.listdir('images'):
        image_path = images_dir + '/' + image
        texts[image] = extract_text_from_image(image_path)