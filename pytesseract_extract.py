# import system modules
import sys
import os
import json

# import external modules
import pytesseract
import cv2
from PIL import Image

""" Module documentation
"""


def remove_boxes_morphological(image_path):
    """ Pretty good
    """
    # Convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Create kernel for morphological operations
    # Horizontal kernel to detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (105, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Vertical kernel to detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 105))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine horizontal and vertical lines to get box structure
    box_mask = cv2.add(horizontal_lines, vertical_lines)
    
    # Remove boxes from original binary image
    result = cv2.subtract(binary, box_mask)
    
    # Convert back to original color space
    result = cv2.bitwise_not(result)
    cv2.imshow('Cleaned', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result


def extract_text_from_image(image_path):
    """ Extracts texts from machinery image. The texts returned are:
        Part Number / Serial Number / Aircraft ID / Engine Number / Time on Wing
    """
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    try:
        image = remove_boxes_morphological(image_path)
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