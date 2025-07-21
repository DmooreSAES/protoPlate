# import internal libraries
import os
import re
import json
import warnings

# import external libraries
import easyocr

""" Module Documentation

In all images "ASSY" is the part number
serial number is either "S/N" or "SER"

extract remove any space, dots
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


class Image:
    "Class representing all actions on Image"

    def __init__(self, path):
        if not os.path.exists(image_path):
            assert(f"File not found: {image_path}")
        self.image_path = path
        self.image_text = self.get_image_text()
        self.part_number = ''
        self.serial_number = ''

    def get_image_text(self):
        """ Extracts texts from machinery image.
        """
        reader = easyocr.Reader(['en'])  # You can add more languages like ['en', 'fr']
        results = reader.readtext(self.image_path,  paragraph=True)

        texts = ''
        for bounding_box, text in results:
            texts += f'{text} \n'
            # print(f"Detected text: {text} (Confidence: {unknown:.2f})")
        return texts.upper()
    
    def get_part_number_and_serial_number(self):
        """ Extracts machine part number, and serial number via regex from the already extracted image text
        """
        match = re.findall('ASSY .+SUBASSY', self.image_text)
        print(match)
        if match:
            match = ''.join(match[0].split(' ')[1:-1])
            part_number, serial_number = match.split('SER')
            self.part_number = part_number
            self.serial_number = serial_number
        else:
            return



if __name__=="__main__":
    images_dir = os.path.abspath('images')
    texts = {}
    for image in os.listdir('images'):
        image_path = images_dir + '/' + image
        image = Image(image_path)
        image.get_part_number_and_serial_number()
        print(image.part_number, image.serial_number)
        # texts[image] = extract_text_from_image(image_path)
    with open('output.json', 'w') as f:
        json.dump(texts, f, indent=3)
