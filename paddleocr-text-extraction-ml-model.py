!pip install paddlepaddle-gpu paddleocr opencv-python-headless

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageEnhance
from io import BytesIO
import re
from paddleocr import PaddleOCR
import logging

# Suppress logs from paddleocr and paddlepaddle
logging.getLogger("ppocr").setLevel(logging.WARNING)
logging.getLogger("paddleocr").setLevel(logging.WARNING)
logging.getLogger("paddle").setLevel(logging.WARNING)
logging.getLogger("paddlehub").setLevel(logging.WARNING)

# Optionally suppress root logger or set it to WARNING or ERROR
logging.getLogger().setLevel(logging.WARNING)

# Unit map for normalization
unit_conversion_map = {
    'g': 'gram', 'ge': 'gram', 'g e': 'gram', 'kg': 'kilogram', 'gram': 'gram', 'kilogram': 'kilogram',
    'ounce': 'ounce', 'oz': 'ounce', 'pound': 'pound', 'lbs': 'pound', 'Ibs': 'pound', 'ibs': 'pound',
    'cm': 'centimetre', 'centimetre': 'centimetre', 'meter': 'metre', 'm': 'metre', 'millimetre': 'millimetre',
    'mm': 'millimetre', "'": 'foot', 'foot': 'foot', 'ft': 'foot', 'inch': 'inch', '"': 'inch', 'in': 'inch',
    'yard': 'yard', 'yd': 'yard', 'ton': 'ton', 'microgram': 'microgram', 'µg': 'microgram', 'milligram': 'milligram',
    'mg': 'milligram', 'kilovolt': 'kilovolt', 'kV': 'kilovolt', 'volt': 'volt', 'v': 'volt', 'watt': 'watt', 
    'watts': 'watt', 'W': 'watt', 'litre': 'litre', 'liter': 'litre', 'l': 'litre', 'ml': 'millilitre', 
    'millilitre': 'millilitre', 'centilitre': 'centilitre', 'cL': 'centilitre', 'cubic foot': 'cubic foot', 
    'ft³': 'cubic foot', 'ft3': 'cubic foot', 'cubic inch': 'cubic inch', 'in³': 'cubic inch', 'in3': 'cubic inch', 
    'cup': 'cup', 'decilitre': 'decilitre', 'dL': 'decilitre', 'fluid ounce': 'fluid ounce', 'fl oz': 'fluid ounce', 
    'gallon': 'gallon', 'gal': 'gallon', 'imperial gallon': 'imperial gallon', 'imp gal': 'imperial gallon',
    'microlitre': 'microlitre', 'µL': 'microlitre', 'pint': 'pint', 'pt': 'pint', 'quart': 'quart', 'qt': 'quart'
}
# Entity to unit map for allowed units
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 
                    'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

allowed_units = {unit for entity in entity_unit_map for unit in entity_unit_map[entity]}
# Image enhancement function
def enhance_image(image_np, scale_factor=2.0, sharpness_factor=2.0, output_size=(1024, 1024)):
    pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Sharpness(pil_image)
    enhanced_image = enhancer.enhance(sharpness_factor)
    new_size = (int(pil_image.width * scale_factor), int(pil_image.height * scale_factor))
    resized_image = enhanced_image.resize(new_size, Image.Resampling.LANCZOS)
    image_cv = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
    final_image = cv2.resize(image_cv, output_size, interpolation=cv2.INTER_CUBIC)
    return final_image

# OCR text detection (handling 'O' and 'o' as '0')
def detect_text_from_image(image):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    results = ocr.ocr(image)
   
    # Check if OCR results are empty or None
    if not results or results[0] is None:
        return ""  # Return empty string if no text is detected
   
    text_data = ' '.join([line[1][0] for line in results[0]])
    text_data = re.sub(r'[Oo]', '0', text_data)  # Replace 'O' and 'o' with '0'
    return text_data

# Entity extraction regular expressions
ENTITY_REGEX = {
    'item_weight': r'(\d+\.?\d*)\s?(g|kg|gram|kilogram|ounce|oz|pound|lbs|ibs|Ibs|microgram|µg|milligram|mg|ton)',
    'maximum_weight_recommendation': r'(\d+\.?\d*)\s?(g|kg|gram|kilogram|ounce|oz|pound|lbs|microgram|µg|milligram|mg|ton)',
    'item_volume': r'(\d+\.?\d*)\s?(ml|millilitre|liter|litre|centilitre|cL|cup|decilitre|dL|fluid ounce|fl oz|gallon|gal|imperial gallon|imp gal|microlitre|µL|pint|pt|quart|qt|cubic foot|ft³|cubic inch|in³)',
    'voltage': r'(\d+\.?\d*)\s?(v|volt|kilovolt|kV|millivolt)',
    'wattage': r'(\d+\.?\d*)\s?(w|watt|kilowatt|W)',
    'width': r'(\d+\.?\d*)\s?(cm|centimetre|meter|metre|mm|foot|ft|inch|in|"|yard|yd)',
    'height': r'(\d+\.?\d*)\s?(cm|centimetre|meter|metre|mm|foot|ft|inch|in|"|yard|yd)',
    'depth': r'(\d+\.?\d*)\s?(cm|centimetre|meter|metre|mm|foot|ft|inch|in|"|yard|yd)'
}
def extract_entity_values(text):
    all_matches = []
    for entity_name, pattern in ENTITY_REGEX.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            all_matches.append((entity_name, matches))
    return all_matches

def normalize_unit(unit):
    return unit_conversion_map.get(unit.lower(), unit)

# Extract entity value based on text
def extract_entity_value(text, entity_name):
    all_entities = extract_entity_values(text)
    
    for entity, matches in all_entities:
        if entity == entity_name:
            value, unit = matches[0]
            normalized_unit = normalize_unit(unit)
            if normalized_unit in allowed_units:
                return f"{value} {normalized_unit}"
    
    return ""

# Process the dataset
def process_dataset(csv_path):
    df = pd.read_csv(csv_path)
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

    predictions = []

    for idx, row in df.iterrows():
        image_url = row['image_link']
        entity_name = row['entity_name']
        
        try:
            response = requests.get(image_url, timeout=10)
            img_array = np.array(Image.open(BytesIO(response.content)))
            enhanced_image = enhance_image(img_array)
            detected_text = detect_text_from_image(enhanced_image)
            predicted_value = extract_entity_value(detected_text, entity_name)
            predictions.append(predicted_value.strip())
        except Exception as e:
            print(f"Error processing image at {image_url}: {e}")
            predictions.append('')

    # Create DataFrame for output
    output_df = pd.DataFrame({
        'index': df.index,
        'predicted_value': predictions
    })

    # Save output to CSV
    output_file_path = '/kaggle/working/predictions_output.csv'
    output_df.to_csv(output_file_path, index=False)

    print(f"Output CSV saved to {output_file_path}")

# Path to the dataset
csv_path = '/kaggle/input/test-datasets/test_11.csv'

# Run the processing
process_dataset(csv_path)
