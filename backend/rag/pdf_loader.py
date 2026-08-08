"""
Extract text from all PDFs.
"""

from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def load_pdf_folder(folder_path):

    folder = Path(folder_path)

    all_text = ""

    pdf_files = sorted(folder.glob("*.pdf"))

    for pdf in pdf_files:

        print(f"Processing {pdf.name}")

        all_text += extract_text_from_pdf(pdf)

        all_text += "\n"

    print()

    print("Finished reading PDFs.")

    return all_text
