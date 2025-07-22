
import io
import os
import re
import json

from google.cloud import vision

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/ibrahimaderinto/Desktop/Upwork/Dennis Moore/protoPlate/visionAI Keys.json'

def detect_text(image_path):
    """Detects text in an Image file."""
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # texts_ = ''
    # for text in texts:
    #     print(text.description)
    #     texts_ += f'{text.description}  '

    if response.error.message:
        raise Exception(f'{response.error.message}')
    
    return re.sub('\n', ' ', texts[0].description)


if __name__=="__main__":
    images_dir = os.path.abspath('images')
    texts = {}
    for image in os.listdir('images'):
        image_path = images_dir + '/' + image
        texts[image] = detect_text(image_path)
    with open('output.json', 'w') as f:
        json.dump(texts, f, indent=3)