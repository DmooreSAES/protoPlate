# import internal libraries
import os

# import external libraries
import cv2
import numpy as np


class Preprocess:
    """ For preprocessing image for better OCR result.
    """

    def __init__(self, image_path):
        """ Instantiates the object

            Arguments:
                image_path - path to the image to preprocess
        """
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.image = self.resize()
        self.image = self.crop_largest_contour()

    def resize(self) -> object:
        """ Resize if too large (Vision API works best with images < 4MB)
        """
        height, width = self.image.shape[:2]
        if max(height, width) > 2048:
            scale = 2048 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            return img
        else:
            return self.image

    def crop_largest_contour(self, output_path=None) -> object:
        """
        Find the largest contour in an image and crop it out.
        
        Args:
            output_path (str): Path to save cropped image (optional)
        
        Returns:
            cropped Image
        """
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
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
        cropped_img = self.image[y:y+h, x:x+w]
        
        # Save the cropped image if output path is provided
        if output_path:
            cv2.imwrite(output_path, cropped_img)
            print(f"Cropped image saved to: {output_path}")
        
        # return cropped_img, (x, y, w, h), largest_contour
        return cropped_img

    def invert(self) -> object:
        """ Inverts an image. An inverted image is like a photographic negative, where black pixels, 
            or anything close becomes white, and white becomes black. This updated the image to it's
            dilated version.

            Returns:
                dilated image
        """
        inverted_image = cv2.bitwise_not(self.image)
        self.image = inverted_image
        return inverted_image

    def gray(self) -> object:
        """ Returns the gray version of an image. This updated the image to it's grayed verison.
        """
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = gray_image
        return gray_image

    def binarize(self) -> object:
        """ Binarizes an image. To do this, the image is first converted to grayscale. Grayscale refers to 
            scalling all colors to somewhere between white and black. Then the image is binarized.
            In a binarized image, only completely black and white pixels will remain. The value of the 
            threshold value determines how much color is converted to white or black, between 0 (black) - 
            255 (white). This updated the image to it's binarized version.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # If pixel > threshold (first digit) 210, it becomes maxval 230
        # If pixel <= threshold, it becomes 0.
        # Meaning those very close to white, turn it to white, else turn it to black.
        # More white = less pixel, and threshold value, more black = more pixel, threshold values.
        thresh, binary_image = cv2.threshold(gray, 120, 230, cv2.THRESH_BINARY)
        self.image = binary_image
        return binary_image

    def erode(self) -> object:
        """ Refers to watering down the font tickness of text, that is making it thinner.
            Image should contain little to no noise here.
            It's commonly used to shrink or thin objects in an image, remove small noise, 
            or disconnect narrow bridges between objects. This updated the image to it's
            eroded verison.
        """
        image = cv2.bitwise_not(self.image)
        kernel = np.ones((2,2),np.uint8)
        eroded_image = cv2.erode(image, kernel, iterations=1)
        eroded_image = cv2.bitwise_not(image)
        self.image = eroded_image
        return eroded_image

    def dilate(self) -> object:
        """ Refers to strenghtening the font tickness of texts in an image, that is making it thicker.
            Image should contain little to no noise here. This updated the image to it's dilated version.
            When to use dilation:
                Filling small gaps in text characters
                Connecting broken text elements
                Making thin text more readable
        """
        image = cv2.bitwise_not(self.image)
        kernel = np.ones((2,2),np.uint8)
        dilated_image = cv2.dilate(image, kernel, iterations=1)
        dilated_image = cv2.bitwise_not(image)
        self.image = dilated_image
        return dilated_image
    
    def remove_boxes_morphological(self) -> object:
        """ Remove boxes in image with the morphological approach. The image should be in gray form.
            Slithly good

            Returns:
                The resulting image
        """
        # Convert to grayscale
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold
        _, binary = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
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
        self.image = result
        return result



if __name__=="__main__":
    images_dir = os.path.abspath('images')
    texts = {}
    for image in os.listdir('images'):
        print(image)
        image_path = images_dir + '/' + image
        img = Preprocess(image_path)
        img = img.gray()
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()