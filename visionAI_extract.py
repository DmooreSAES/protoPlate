# inertnal modules
import io
import os
import re
import json

# external modules
from google.cloud import vision


""" Module Documentation
"""


class ImageExtract:
    "Class representing all actions on Image"

    def __init__(self, path):
        if not os.path.exists(image_path):
            assert(f"File not found: {image_path}")
        self.image_path = path
        self.image_text = self.get_image_text()
        self.part_number, self.serial_number = self.get_part_and_serial_numbers()

    def get_image_text(self, method='text') -> str:
        """ Extract all text from image at once.
            
            Args:
                method (str): 'text' for text_detection or 'document' for document_text_detection
            
            Returns:
                str: All detected text
        """
        client = vision.ImageAnnotatorClient()

        try:
            with io.open(self.image_path, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)
            
            # document method is useful for better dense text
            if method == 'document':
                response = client.document_text_detection(image=image)
                if response.full_text_annotation:
                    return re.sub('\n', ' ', response.full_text_annotation.text.strip())
                
            elif method=='text':
                response = client.text_detection(image=image)
                if response.text_annotations:
                    return re.sub('\n', ' ', response.text_annotations[0].description.strip())
            else:
                return "Method not recognized"
            
            return "No text detected"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_part_and_serial_numbers(self) -> list:
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
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/ibrahimaderinto/Desktop/Upwork/Dennis Moore/protoPlate/visionAI Keys.json'
    images_dir = os.path.abspath('images')
    texts = {}
    for image in os.listdir('images'):
        print(image)
        image_path = images_dir + '/' + image
        image_extract = ImageExtract(image_path)
        texts[image] = {
                        # 'text': image_extract.image_text,
                        'part number': image_extract.part_number,
                        'serial nnumber': image_extract.serial_number
                        }
    with open('output.json', 'w') as f:
        json.dump(texts, f, indent=3)