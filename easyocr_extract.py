# import internal libraries
import os
import re
import time
import json
import warnings

# import external libraries
import cv2
import easyocr
import numpy as np

""" Module Documentation
"""

warnings.filterwarnings('ignore')

def extract_text_from_image(image_path):
    """ Extracts texts from machinery image. The texts returned are:
        Part Number / Serial Number / Aircraft ID / Engine Number / Time on Wing
    """
    reader = easyocr.Reader(['en'])  # You can add more languages like ['en', 'fr']
    results = reader.readtext(image_path,  paragraph=True)

    texts = ''
    for bounding_box, text in results:
        texts += f'{text} \n'
        # print(f"Detected text: {text} (Confidence: {unknown:.2f})")
    return texts


class ImageExtract:
    "Class representing all actions on Image"

    def __init__(self, path):
        if not os.path.exists(image_path):
            assert(f"File not found: {image_path}")
        self.image_path = path 
        # self.image = self.preprocess_image()
        self.image_text = self.get_image_text()
        self.part_number, self.serial_number = self.get_part_and_serial_numbers()

    def preprocess_image(self):
        """ Preprocess image to allow better text extraction
        """
        image = cv2.imread(self.image_path)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # blurred = cv2.GaussianBlur(gray, (3, 3), 0) # Not good
        # cv2.imshow('Blurred', gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return image


    def get_image_text(self):
        """ Extracts texts from machinery image.
        """
        reader = easyocr.Reader(['en'])  # You can add more languages like ['en', 'fr']
        results = reader.readtext(self.image_path,
                                  allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                                  paragraph=True,
                                #   decoder='beamsearch'      # this increases computational/speed cost
                                )

        texts = ''
        for bounding_box, text in results:
            texts += f'{text} \n'
            # print(f"Detected text: {text} (Confidence: {unknown:.2f})")
        return texts.upper()
    
    def get_part_and_serial_numbers(self):
        """ Extracts machine part number, and serial number via regex from the already extracted image text
        """
        match = re.findall('ASSY.+SUBASSY', self.image_text)
        print(match)
        if match:
            try:
                part_number, serial_number = ['', '']
                match = ''.join(match[0].split(' ')[1:-1])
                part_number, serial_number = match.split('SER')
                serial_number = re.findall('[0-9]{4}[0-9]*', serial_number)[0]
            except Exception as err: pass
            return [part_number, serial_number]
        else:
            return ['', '']



def remove_boxes_morphological(image_path):
    """ Slithly good
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

def remove_boxes_contours(image_path):
    "Produces unexpected results for now"
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask for boxes
    mask = np.zeros_like(binary)
    
    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Filter for rectangular shapes (likely boxes)
        # Adjust these thresholds based on your specific use case
        if (area > 5000 and  # Minimum area threshold
            0.8 < aspect_ratio < 5.0 and  # Reasonable aspect ratio
            cv2.contourArea(contour) / (w * h) > 0.8):  # Rectangle-like shape
            
            # Draw contour on mask (fill it)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Remove detected boxes
    result = cv2.subtract(binary, mask)
    cv2.imshow('Cleaned', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cv2.bitwise_not(result)


if __name__=="__main__":
    images_dir = os.path.abspath('images')
    texts = {}
    for image in os.listdir('images'):
        print(image)
        image_path = images_dir + '/' + image
        # remove_boxes_morphological(image_path)
        # continue
        image_extract = ImageExtract(image_path)
        texts[image] = {
                        # 'text': image_extract.image_text,
                        'part number': image_extract.part_number,
                        'serial nnumber': image_extract.serial_number
                        }
        # texts[image] = extract_text_from_image(image_path)
    with open('outputs/easyocr output.json', 'w') as f:
        json.dump(texts, f, indent=3)
