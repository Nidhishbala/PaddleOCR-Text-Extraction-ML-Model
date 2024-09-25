# **Product Entity Text Extraction from Images**
## **About the Project**
This project was our submission for the **Amazon ML Challenge 2024**. We secured [209th](https://unstop.com/hackathons/amazon-ml-challenge-amazon-1100713/coding-challenge/200089) place out of around 75,000 participants from all over India and we ranked **9th** across all Vellore Institute of Technology campuses.

It demonstrates an approach to extract key product attributes (such as weight, dimensions, and volume) from images using advanced OCR (Optical Character Recognition) techniques, entity extraction, and unit normalization.
We tested the model with around 130K (1.3 Lakh) images and obtained an F1-Score of 0.443 indicating good performance in predicting product attributes.

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

**NOTE: The datasets contain a large amount of data, and will take anywhere from 4 to 9 hours to successfully compile. Also make sure you turn on GPU T4 X2 in Kaggle while the compilation is in progress.**
## **ML Model Used**
The core machine learning model used in this solution is PaddleOCR, known for its high accuracy in text recognition across various image formats and orientations. Key features include:

- **Angle Detection**: Ensures text at varying angles is correctly identified.
- **Precision and Adaptability**: Handles text in diverse product images, including small or rotated text.

## **Output**
After processing, the results are saved to a CSV file with columns:

- index: The index of the image in the input CSV.
- predicted_value: The predicted value of the entity (e.g., `500 grams`, `12 inches`).

## **Experiment Details**
- **OCR**: PaddleOCR's angle detection ensures text from rotated or skewed images is properly extracted.
- **Unit Normalization**: Extracted units are normalized to match the expected format (e.g., oz to ounce, kg to kilogram).
- **Model**: The solution is based on the PaddleOCR model, optimized with image enhancement techniques.

## **Conclusion**
This project demonstrates how effective image preprocessing and OCR can be for extracting meaningful product information from images, especially in e-commerce and other data-rich industries. It highlights the power of combining advanced image processing, machine learning, and rule-based approaches.  

## **License**
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## **Contact**
- Allen Reji - allenreji@gmail.com
- Nathania Rachael - nathaniarachael@gmail.com
- Nidhish Balasubramanya - nidhishbala3006@gmail.com
- Jacob Cherian - jakecherian10@gmail.com
