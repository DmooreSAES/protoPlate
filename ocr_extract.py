import pytesseract
from PIL import Image
import sys
import os

# Optional: Set the tesseract command path (needed on Windows if not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ocr_extract.py path_to_image")
    else:
        extract_text_from_image(sys.argv[1])
