# Import
import argparse
from pypdf import PdfReader
import json
import os
import pytesseract  # OCR
from pdf2image import convert_from_path  # Convert PDF pages to images


def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments containing `data_dir` and `save_path`.
    """
    parser = argparse.ArgumentParser(
        description="This script is used for compiling the documents into a single JSON file, using the Label Studio format."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="The path to the data folder, containing subfolders with .txt and/or .pdf files.",
        default="./data/raw/",
        required=False,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="The path for saving the final JSON dataset.",
        default="./data/dataset.json",
        required=False,
    )

    return parser.parse_args()


def is_pdf_readable(pdf_path):
    """
    Check if a PDF contains selectable text.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        bool: True if the PDF contains selectable text, False otherwise.
    """
    doc = PdfReader(pdf_path)
    pages = doc.pages
    text = ""
    for page in pages:
        text += page.extract_text()
    return bool(text.strip())  # Return True if text is found, False otherwise


def process_pdf(pdf_path):
    """
    Process a PDF file to extract text. If the PDF contains selectable text, it is extracted.
    Otherwise, OCR is applied to images of the PDF pages.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    if is_pdf_readable(pdf_path):
        doc = PdfReader(pdf_path)
        pages = doc.pages
        text = ""
        for page in pages:
            text += page.extract_text()

        return text

    else:
        images = convert_from_path(pdf_path)  # Convert PDF pages to images
        text = "\n".join([pytesseract.image_to_string(img) for img in images])

        return text


def remove_returns(text):
    """
    Replace carriage return and newline characters with newline characters.

    Args:
        text (str): Input text.

    Returns:
        str: Text with carriage returns replaced.
    """
    text = text.replace("\r\n", "\n")
    return text


def process_data(data_folder):
    """
    Process all text and PDF files in the specified data folder and compile them into a dataset.

    Args:
        data_folder (str): Path to the folder containing subfolders with .txt and/or .pdf files.

    Returns:
        list: A list of dictionaries representing the dataset in Label Studio format.
    """
    data_list = []  # Insert entry_dict(s)

    dirs = [f.name for f in os.scandir(data_folder) if f.is_dir()]

    for dir in dirs:

        dir_path = os.path.join(data_folder, dir)
        files = [f.name for f in os.scandir(dir_path) if f.is_file()]

        for file in files:

            file_path = os.path.join(dir_path, file)

            if file.endswith(".txt"):

                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

            if file.endswith(".pdf"):

                text = process_pdf(file_path)

            text = remove_returns(text)

            data_dict = {
                "text": "text",
                "source_dataset": "source_dataset",
                "file_name": file,
            }

            # Insert values from data
            data_dict["text"] = text
            data_dict["source_dataset"] = dir

            # Insert into the entry dict for this data point
            entry_dict = {"data": data_dict, "predictions": []}

            # Finally, append to the data list
            data_list.append(entry_dict)

            print(f"Processed document: {file}")

    return data_list


def save_json(data, save_path):
    """
    Save the dataset to a JSON file.

    Args:
        data (list): The dataset to save.
        save_path (str): Path to save the JSON file.
    """
    json_object = json.dumps(data, indent=2)

    with open(save_path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)


def main():
    """
    Main function to parse arguments, process data, and save the dataset.
    """
    args = parse_arguments()
    data_list = process_data(args.data_dir)
    save_json(data_list, args.save_path)


if __name__ == "__main__":
    main()
