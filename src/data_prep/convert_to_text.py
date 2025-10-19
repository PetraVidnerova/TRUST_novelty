import os
import re
import tqdm
import pdfplumber

def clean_text(text):
    # remove page numbers 
    text = re.sub(r'^\s*(Page\s*)?\d+\s*$', '', text, flags=re.MULTILINE)
    # remove multiple empty lines
    text = re.sub(r'\n{2,}', '\n\n', text)
    # remove splitted words (hyphen at linebreak)
    text = re.sub(r'-\n', '', text)
    # join lines
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    return text.strip()

N = 5011 
not_converted = {652} # ended with error

with open("withdrawn.txt", "r") as f:
    lines = f.readlines()
    withdrawn_ids = list(map(int, lines))

for i in tqdm.tqdm(range(N)):
    if i in withdrawn_ids:
        continue
    if i in not_converted:
        continue

    if os.path.exists(f"TXT2/{i}.txt"):
        print(f"{i}:  exists, skip")
        continue


    filename = f"PDF/{i}.pdf"    
    
    full_text = ""
    with pdfplumber.open(filename) as pdf:
        for page, text in enumerate(pdf.pages):
            page_text = text.extract_text()
            if not page_text:
                continue

            lines = page_text.split('\n')
            if len(lines) > 6:
                lines = lines[2:-2]  # remove headers and footers 
            page_text = "\n".join(lines)

            clean_page = clean_text(page_text)
            full_text += clean_page + "\n"
            
    with open(f"TXT2/{i}.txt", "w") as f:
        print(full_text, file=f)
