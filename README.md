# **Product Entity Text Extraction from Images**
This project was our submission to the Amazon ML Challenge 2024. We secured 209th place out of around 75,000 participants from all over India and we ranked 9th across all Vellore Institute of Technology campuses.

It demonstrates an approach to extract key product attributes (such as weight, dimensions, and volume) from images using advanced OCR (Optical Character Recognition) techniques, entity extraction, and unit normalization.

## **Overview**
The solution focuses on:

1.  Enhancing image quality for OCR.
2.  Using PaddleOCR for text detection and recognition.
3.  Extracting product entities (like weight, volume, dimensions) using regular expressions.
4.  Normalizing the extracted units to ensure consistency.
## **Features**
- **Image Preprocessing**: Sharpness adjustment, rescaling, and resizing to improve OCR accuracy.
- **OCR Detection**: PaddleOCR is used to extract text from images, ensuring accurate text detection even from complex images.
- **Entity Extraction**: Regular expressions are employed to extract product attributes such as weight, height, width, depth, voltage, and volume.
- **Unit Normalization**: The extracted units are standardized using a predefined unit conversion map.
## **Dataset**
The input dataset is a CSV file containing image URLs and expected entity names (e.g., `item_weight`, `item_volume`,`wattage` etc.). The script processes each image, extracts relevant text, and matches it to the specified entity. The datasets we used for the final submission are placed inside the `test_datasets` folder. 

**NOTE: The datasets contain a large amount of data, and will take anywhere from 4 to 9 hours to successfully compile. Also make sure you turn on GPU T4 X2 while the compilation is in progress.**
