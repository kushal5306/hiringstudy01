import subprocess
import sys

# Function to install the required packages, if this does not do the job, please install via terminal requirements.txt
def install_requirements():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

# Ensure the necessary packages are installed
install_requirements()

# Now, import the required packages
from typing import Dict, List
import fitz  # PyMuPDF
import re
from PIL import Image
import numpy as np
import pytesseract
from pytesseract import Output

# Set the path to the Tesseract executable if it's not in your PATH, check solution_readme.txt file on how to install tesseract ocr on your machine, the code wont work otherwise
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_features(page, page_image) -> dict:
    """
    Extract visual and textual features from a PDF page.
    
    Args:
        page (fitz.Page): The PDF page object.
        page_image (PIL.Image.Image): The image representation of the PDF page.
    
    Returns:
        dict: A dictionary containing extracted text, watermark text, and border colors.
    """
    features = {}
    text = page.get_text("text")
    features['text'] = text
    watermark_text = extract_watermark_text(page_image)
    features['watermark'] = watermark_text
    border_colors = analyze_border_color(page_image)
    features['border_colors'] = border_colors
    return features

def extract_watermark_text(image) -> str:
    """
    Extract watermark text from a given PIL image using OCR.
    
    Args:
        image (PIL.Image.Image): The image representation of the PDF page.
    
    Returns:
        str: The extracted watermark text.
    """
    grayscale_image = image.convert('L')
    ocr_result = pytesseract.image_to_string(grayscale_image, config='--psm 6', output_type=Output.STRING)
    return ocr_result.strip()

def analyze_border_color(image) -> List[tuple]:
    """
    Analyze the border color of a given PIL image.
    
    Args:
        image (PIL.Image.Image): The image representation of the PDF page.
    
    Returns:
        List[tuple]: A list of border colors (top, bottom, left, right).
    """
    border_colors = []
    np_image = np.array(image)
    top_color = np.mean(np_image[0:10, :, :], axis=(0, 1))
    bottom_color = np.mean(np_image[-10:, :, :], axis=(0, 1))
    left_color = np.mean(np_image[:, 0:10, :], axis=(0, 1))
    right_color = np.mean(np_image[:, -10:, :], axis=(0, 1))
    border_colors.extend([tuple(top_color), tuple(bottom_color), tuple(left_color), tuple(right_color)])
    return border_colors

def analyze_keywords(features_list: List[dict], num_pages: int) -> Dict[str, List[int]]:
    """
    Analyze a list of features dictionaries to determine document partitions based on keywords.
    
    Args:
        features_list (List[dict]): A list of dictionaries containing extracted features.
        num_pages (int): The total number of pages in the PDF document.
    
    Returns:
        Dict[str, List[int]]: A dictionary mapping document names to page numbers.
    """
    document_groups = {}
    doc_pattern = re.compile(r'Document (\d+) - Page (\d+)')  
    for i, features in enumerate(features_list):
        text = features['text']
        match = doc_pattern.search(text)
        if match:
            doc_id = match.group(1)
            doc_name = f"Document {doc_id}"
            if doc_name not in document_groups:
                document_groups[doc_name] = []
            document_groups[doc_name].append(i + 1)
    return document_groups

def analyze_watermarks(features_list: List[dict], num_pages: int) -> Dict[str, List[int]]:
    """
    Analyze a list of features dictionaries to determine document partitions based on watermarks.
    
    Args:
        features_list (List[dict]): A list of dictionaries containing extracted features.
        num_pages (int): The total number of pages in the PDF document.
    
    Returns:
        Dict[str, List[int]]: A dictionary mapping document names to page numbers.
    """
    document_groups = {}
    current_document = "Document 1"
    document_groups[current_document] = []
    previous_watermark = None
    for i, features in enumerate(features_list):
        watermark_text = features.get('watermark', "")
        if previous_watermark is not None and watermark_text != previous_watermark:
            current_document = f"Document {len(document_groups) + 1}"
            document_groups[current_document] = []
        document_groups[current_document].append(i + 1)
        previous_watermark = watermark_text
    return document_groups

def analyze_border_colors(features_list: List[dict], num_pages: int) -> Dict[str, List[int]]:
    """
    Analyze a list of features dictionaries to determine document partitions based on border colors.
    
    Args:
        features_list (List[dict]): A list of dictionaries containing extracted features.
        num_pages (int): The total number of pages in the PDF document.
    
    Returns:
        Dict[str, List[int]]: A dictionary mapping document names to page numbers.
    """
    document_groups = {}
    current_document = "Document 1"
    document_groups[current_document] = []
    previous_colors = None
    for i, features in enumerate(features_list):
        border_colors = features['border_colors']
        if previous_colors is not None and border_colors != previous_colors:
            current_document = f"Document {len(document_groups) + 1}"
            document_groups[current_document] = []
        document_groups[current_document].append(i + 1)
        previous_colors = border_colors
    return document_groups

def partition_the_pdf_document(input_pdf: str) -> Dict[str, List[int]]:
    """
    Partition a PDF document based on various features (keywords, watermarks, border colors).
    
    Args:
        input_pdf (str): The path to the input PDF document.
    
    Returns:
        Dict[str, List[int]]: A dictionary mapping document names to page numbers.
    
    Raises:
        ValueError: If no strategy can be applied as features are not found in all pages.
    """
    reader = fitz.open(input_pdf)
    num_pages = reader.page_count
    features_list = []
    for i in range(num_pages):
        page = reader.load_page(i)
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        features = extract_features(page, image)
        features_list.append(features)

    # Analyze features to determine partitions based on keywords
    document_groups = analyze_keywords(features_list, num_pages)
    if document_groups:
        return document_groups

    # If no partitions are found based on keywords, analyze features based on watermarks
    document_groups = analyze_watermarks(features_list, num_pages)
    if document_groups:
        return document_groups

    # If no partitions are found based on watermarks, analyze features based on border colors
    document_groups = analyze_border_colors(features_list, num_pages)
    if document_groups:
        return document_groups

    # If no partitions are found based on any features
    raise ValueError("No strategy can be applied as features are not found in all pages")

# Usage
input_pdf: str = "grouped_documents.pdf"
try:
    partitions: Dict[str, List[int]] = partition_the_pdf_document(input_pdf)
    print(f"Partitions: {partitions}")
except ValueError as e:
    print(e)
