# import internal libraries
import os
import json

# import external libraries
import easyocr

"""
"""

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


if __name__=="__main__":
    images_dir = os.path.abspath('images')
    texts = {}
    for image in os.listdir('images'):
        image_path = images_dir + '/' + image
        texts[image] = extract_text_from_image(image_path)
    with open('output.json', 'w') as f:
        json.dump(texts, f, indent=3)
