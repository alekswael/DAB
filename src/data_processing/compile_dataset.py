# Import
import argparse
from pypdf import PdfReader
import json
import os
import pytesseract  # OCR
from pdf2image import convert_from_path  # Convert PDF pages to images

def parse_arguments():
    parser = argparse.ArgumentParser(description="This script is used for compiling the documents into a single JSON file, using the Label Studio format.")
    parser.add_argument(
        '-d', '--data_path',
        type=str, 
        help='The path to the data folder, containing subfolders with .txt and/or .pdf files.',
        default='./data/raw/',
        required=False
    )
    parser.add_argument(
        '-s', '--save_path', 
        type=str, 
        help='The path for saving the final JSON dataset.',
        default="./data/DAB_dataset.json",
        required=False
    )

    return parser.parse_args()

def is_digitally_born(pdf_path):
    """Check if a PDF contains selectable text."""
    doc = PdfReader(pdf_path)
    pages = doc.pages
    text = ""
    for page in pages:
        text += page.extract_text()
    return bool(text.strip())  # Return True if text is found, False otherwise

def process_pdf(pdf_path):
    """Process the PDF based on whether it contains text or is scanned."""
    if is_digitally_born(pdf_path):

        #print(f"{pdf_path}: Digitally born - Extracting text with PdfReader")
        
        doc = PdfReader(pdf_path)
        pages = doc.pages
        text = ""
        for page in pages:
            text += page.extract_text()
        
        return text

    else:

        #print(f"{pdf_path}: Scanned PDF - Applying OCR")

        images = convert_from_path(pdf_path)  # Convert PDF pages to images
        text = "\n".join([pytesseract.image_to_string(img) for img in images])

        return text
    
def remove_returns(text):
    text = text.replace("\r\n", "\n")
    return text

def process_data(data_folder):

    data_list = [] # Insert entry_dict(s)

    dirs = [f.name for f in os.scandir(data_folder) if f.is_dir()]

    for dir in dirs:

        dir_path = os.path.join(data_folder, dir)
        files = [f.name for f in os.scandir(dir_path) if f.is_file()]

        for file in files:

            file_path = os.path.join(dir_path, file)

            if file.endswith(".txt"):

                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            if file.endswith(".pdf"):
                
                text = process_pdf(file_path)
            
            text = remove_returns(text)

            data_dict = {
                "text": "text",
                "source_dataset": "source_dataset",
                "file_name": file
            }
            
            # Insert values from data
            data_dict["text"] = text
            data_dict["source_dataset"] = dir

            # Insert into the entry dict for this data point
            entry_dict = {
                "data": data_dict,
                "predictions": []
            }

            # Finally, append to the data list
            data_list.append(entry_dict)

            print(f"Processed document: {file}")

    return data_list

def save_json(data, save_path):

    json_object = json.dumps(data, indent=2)
    
    with open(save_path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)

def main():
    args = parse_arguments()
    data_list = process_data(args.data_path)
    save_json(data_list, args.save_path)

if __name__ == "__main__":
    main()