# Product Entity Extraction from Images
This project demonstrates an approach to extract key product attributes (such as weight, dimensions, and volume) from images using advanced OCR (Optical Character Recognition) techniques, entity extraction, and unit normalization.

Overview
The solution focuses on:

Enhancing image quality for OCR.
Using PaddleOCR for text detection and recognition.
Extracting product entities (like weight, volume, dimensions) using regular expressions.
Normalizing the extracted units to ensure consistency.
Features
Image Preprocessing: Sharpness adjustment, rescaling, and resizing to improve OCR accuracy.
OCR Detection: PaddleOCR is used to extract text from images, ensuring accurate text detection even from complex images.
Entity Extraction: Regular expressions are employed to extract product attributes such as weight, height, width, depth, voltage, and volume.
Unit Normalization: The extracted units are standardized using a predefined unit conversion map.

