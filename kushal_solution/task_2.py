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

# Now, import the required packages
import pytesseract
from pdf2image import convert_from_path
import pdfplumber

from pytesseract import Output
from typing import List
import numpy as np
# Set the path to the Tesseract executable if it's not in your PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def classify_all_pages(input_pdf: str) -> List[int]:
    """
    Analyze all pages in the input PDF and determine the class of the pdf page.

    Args:
    input_pdf (str): The file path of the input PDF.

    Returns:
    List[int]: A list of classes for each page. 
            0: machine-readable
            1: non-machine readable but OCR-able
            2: non-machine readable and not OCR-able
    """
    classes = []
    try:
        with pdfplumber.open(input_pdf) as pdf:
            for page_number in range(len(pdf.pages)):
                try:
                    page_class = classify_page(pdf, input_pdf, page_number + 1)
                    classes.append(page_class)
                except Exception as e:
                    print(f"Error processing page {page_number + 1}: {e}")
                    classes.append(2)  # Default to class 2 if there is an error
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return []

    return classes

def classify_page(pdf, input_pdf: str, page_number: int) -> int:
    """
    Determine the class of the pdf page.

    Args:
    pdf (pdfplumber.PDF): The PDF document object.
    input_pdf (str): The file path of the input PDF.
    page_number (int): The number of the page in the PDF.

    Returns:
    int: The page is 
        0: machine-readable
        1: non-machine readable but OCR-able
        2: non-machine readable and not OCR-able
    """
    try:
        pdf_page = pdf.pages[page_number - 1]  # page_number is 1-based in pdfplumber
        # Crop the page to remove header and footer
        width = pdf_page.width
        height = pdf_page.height
        # Adjust these values as necessary to remove header and footer (in points)
        # if i didnt remove header and footer all pages will be classified as machine readable for given pdf.
        top_margin = 50  # Points
        bottom_margin = 50  # Points
        cropped_page = pdf_page.within_bbox((0, top_margin, width, height - bottom_margin))
        text = cropped_page.extract_text()
        if text and text.strip():
            return 0

        # Convert page to image
        images = convert_from_path(input_pdf, first_page=page_number, last_page=page_number)

        if images:
            image = images[0]
            # Perform OCR on the cropped image
            image_np = np.array(image)
            # Crop the image to remove header and footer (in pixels)
            # if i didnt remove header and footer all pages will be classified as machine readable.
            top_margin_px = 50  # Pixels
            bottom_margin_px = 50  # Pixels
            cropped_image = image_np[top_margin_px:image_np.shape[0] - bottom_margin_px, :]
            # Perform OCR on the cropped image using Tesseract
            ocr_result = pytesseract.image_to_string(cropped_image, output_type=Output.STRING)
            if ocr_result and ocr_result.strip():
                return 1

        # If no text is found, classify as non-machine readable and not OCR-able
        return 2

    except Exception as e:
        print(f"Error processing page {page_number}: {e}")
        return 2

# Usage
input_pdf = "grouped_documents.pdf"
page_classes = classify_all_pages(input_pdf)
print(f"Classes for each page: {page_classes}")
