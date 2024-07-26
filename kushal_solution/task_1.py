import subprocess
import sys

def install_requirements(requirements_file='requirements.txt'):
    """
    Install the packages specified in the requirements file.

    Args:
        requirements_file (str): Path to the requirements.txt file. Default is 'requirements.txt'.
    """
    try:
        with open(requirements_file, 'r') as file:
            requirements = file.read().splitlines()
        for requirement in requirements:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', requirement])
        print("All requirements installed successfully.")
    except Exception as e:
        print(f"An error occurred while installing requirements: {e}")

# Install required packages
install_requirements()

import numpy as np
import cv2
from skimage.transform import radon
import fitz  # PyMuPDF
from PIL import Image

def process_image(image):
    """
    Determine the rotation angle required to make the image upright.

    Args:
        image (PIL.Image.Image): The image to process.

    Returns:
        float: The angle in degrees to rotate the image so that it is upright.
               The angle is between 0 and 180 degrees.
    """
    I = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    I = I - np.mean(I)  # Normalize the image brightness
    sinogram = radon(I)
    r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
    rotation = np.argmax(r)
    return 180 - rotation  # Compute the rotation angle

def rotate_image(image, angle):
    """
    Rotate the given image by the specified angle.

    Args:
        image (PIL.Image.Image): The image to rotate.
        angle (float): The angle in degrees to rotate the image.

    Returns:
        PIL.Image.Image: The rotated image.
    """
    img = np.array(image)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    dst = cv2.warpAffine(img, M, (w, h))
    return Image.fromarray(dst)

def needs_deskew(image, threshold=10):
    """
    Check if the image needs deskewing based on the standard deviation threshold.

    Args:
        image (PIL.Image.Image): The image to check.
        threshold (float): The threshold for standard deviation to decide if deskewing is needed.

    Returns:
        bool: True if the standard deviation exceeds the threshold, indicating the image may need deskewing.
    """
    I = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    I = I - np.mean(I)  # Normalize the image brightness
    sinogram = radon(I)
    r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
    std_dev = np.std(r)
    return std_dev > threshold

def rotate_all_pages_upright(input_pdf, threshold=10):
    """
    Process the PDF to determine the rotation angles for each page and save the rotated images.

    Args:
        input_pdf (str): The file path of the input PDF.
        threshold (float): The standard deviation threshold for detecting skew.

    Returns:
        list of float: A list of rotation angles in degrees for each page in the PDF.
                       Returns 0 if no rotation is needed.
    """
    document = fitz.open(input_pdf)
    angles = []
    rotated_images = []

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        try:
            if needs_deskew(img, threshold):
                angle = process_image(img)
                angles.append(angle)
                rotated_img = rotate_image(img, angle)
                rotated_images.append(rotated_img)
            else:
                angles.append(0)  # No deskewing needed
                rotated_images.append(img)
        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
            angles.append(None)

    if rotated_images:
        rotated_images[0].save('rotated_output.pdf', save_all=True, append_images=rotated_images[1:])
    else:
        print("No images were processed correctly. Output PDF not created.")

    return angles

# Usage
input_pdf = 'grouped_documents.pdf'
angles = rotate_all_pages_uprightf(input_pdf, threshold=10)
print("Angles of rotation for each page:", angles)

    """
      Rotation logic

        This script converts each page into image and then to grayscale and normalizes its brightness. It then applies
        the Radon transform to obtain a sinogram, which represents the projections of the image at
        various angles. By calculating the root mean square (RMS) of these projections, the function
        identifies the angle with the maximum projection strength, indicating the alignment of the
        image content. The required rotation angle to correct the skew is computed as `180 - rotation`,
        where `rotation` is the angle with the highest projection strength. This angle is then used
        to rotate the image so that it is properly aligned. I tried to make as many page as upright possible irrespective of its content,
        as task mentioned to make the page upright and not content. However both would have been ideal but not sure if its possible for this pdf as its randomly generated. 
        I tried looked up many papers and did research also tried to implement hough line transform but did not gave good result. I generated an output of this script with the given 
        pdf its named as rotated_output.pdf for reference. 

    """