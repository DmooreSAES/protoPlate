# import internal libraries
import os
import re
import json
import warnings

# import external libraries
import cv2
import easyocr
import numpy as np
from preprocess import Preprocess

""" Module Documentation
"""

warnings.filterwarnings('ignore')

class ImageExtract:
    "Class representing all text removal actions on Image"

    def __init__(self, path):
        if not os.path.exists(image_path):
            assert(f"File not found: {image_path}")
        self.image_path = path 
        image = Preprocess(self.image_path)
        image.gray()        # Turn image to gray version. improves accuracy for some bad image
        self.image = image.remove_boxes_morphological()
        self.image_text = self.get_image_text()
        self.part_number, self.serial_number = self.get_part_and_serial_numbers()

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



if __name__=="__main__":
    images_dir = os.path.abspath('images')
    texts = {}
    for image in os.listdir('images'):
        print(image)
        image_path = images_dir + '/' + image
        # remove_boxes_contours(image_path)
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
