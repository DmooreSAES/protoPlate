# import internal libraries
import os
import re
import time
import json
import warnings

# import external libraries
import cv2
import numpy as np


def resize(image_path) -> object:
    """ Resize if too large (Vision API works best with images < 4MB)
    """
    # Load image
    img = cv2.imread(image_path)
    
    height, width = img.shape[:2]
    if max(height, width) > 2048:
        scale = 2048 / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return img


def invert(image_path) -> object:
    """ Inverts an image. An inverted image is like a photographic negative, where black pixels, 
        or anything close becomes white, and white becomes black.
    """
    img = cv2.imread(image_path)
    inverted_image = cv2.bitwise_not(img)
    return inverted_image


def gray(image_path) -> object:
    """ Returns the gray version of an image
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def binarize(image_path) -> object:
    """ Binarizes an image. To do this, the image is first converted to grayscale. Grayscale refers to 
        scalling all colors to somewhere between white and black. Then the image is binarized.
        In a binarized image, only completely black and white pixels will remain. The value of the 
        threshold value determines how much color is converted to white or black, between 0 (black) - 
        255 (white),
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # If pixel > threshold (first digit) 210, it becomes maxval 230
    # If pixel <= threshold, it becomes 0.
    # Meaning those very close to white, turn it to white, else turn it to black.
    # More white = less pixel, and threshold value, more black = more pixel, threshold values.
    thresh, binary_image = cv2.threshold(gray, 170, 230, cv2.THRESH_BINARY)
    return binary_image


def largest_contour_cropping(image_path, output_path=None) -> object:
    """
    Find the largest contour in an image and crop it out.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save cropped image (optional)
    
    Returns:
        tuple: (cropped_image, bounding_rect, largest_contour)
    """
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    # You might need to adjust these values based on your image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Alternative: Use adaptive threshold for better results with varying lighting
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the image using the bounding rectangle
    cropped_img = img[y:y+h, x:x+w]
    
    # Save the cropped image if output path is provided
    if output_path:
        cv2.imwrite(output_path, cropped_img)
        print(f"Cropped image saved to: {output_path}")
    
    # return cropped_img, (x, y, w, h), largest_contour
    return cropped_img


def erode(image_path) -> object:
    """ Refers to watering down the font tickness of text, that is making it thinner.
        Image should contain little to no noise here.
        It's commonly used to shrink or thin objects in an image, remove small noise, 
        or disconnect narrow bridges between objects
    """
    img = cv2.imread(image_path)
    image = cv2.bitwise_not(img)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


def dilate(image_path) -> object:
    """ Refers to strenghtening the font tickness of texts in an image, that is making it thicker.
        Image should contain little to no noise here.
        When to use dilation:
            Filling small gaps in text characters
            Connecting broken text elements
            Making thin text more readable
    """
    img = cv2.imread(image_path)
    image = cv2.bitwise_not(img)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image



if __name__=="__main__":
    images_dir = os.path.abspath('images')
    texts = {}
    for image in os.listdir('images'):
        print(image)
        image_path = images_dir + '/' + image
        img = largest_contour_cropping(image_path)
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()