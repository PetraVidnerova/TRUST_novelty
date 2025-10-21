import os
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFSyntaxError

def is_valid_pdf(path):
    try:
        with open(path, "rb") as f:
            parser = PDFParser(f)
            PDFDocument(parser)
        return True
    except (PDFSyntaxError, Exception):
        return False

for filename in os.listdir("../../data/cell/PDF/"):
    if not is_valid_pdf(f"../../data/cell/PDF/{filename}"):
        print(filename)
        os.remove(f"../../data/cell/PDF/{filename}") 
    else:
        print(".", end=" ")
