import os
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFSyntaxError
import tqdm 

def is_valid_pdf(path):
    try:
        with open(path, "rb") as f:
            parser = PDFParser(f)
            PDFDocument(parser)
        return True
    except (PDFSyntaxError, Exception):
        return False

with open("valid_pdfs.txt", "r") as f:
    valid_pdfs = f.readlines()
    valid_pdfs = set([x.strip() for x in valid_pdfs])

filelist = list(os.listdir("../../data/cell/PDF/"))
for filename in tqdm.tqdm(filelist):
    if filename in valid_pdfs:
        continue
    if filename == "bugs":
        continue
    if not is_valid_pdf(f"../../data/cell/PDF/{filename}"):
        print(filename)
        os.rename(f"../../data/cell/PDF/{filename}", f"../../data/cell/PDF/bugs/{filename}") 
    else:
        with open("valid_pdfs.txt", "a") as f:
            print(filename, file=f  )