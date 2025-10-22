import os
import time
import tqdm 
import doi2pdf
from utils.utils import read_df
from alex_utils import get_doi_for_pubmed

if os.path.exists("file_not_found.txt"):
    with open("file_not_found.txt", "r") as f:
        not_found_ids = f.readlines()
        not_found_ids = list(map(lambda x: x.strip(), not_found_ids))
else:
    not_found_ids = []

df = read_df("../../data/novelpy/novelpy_data_test.feather")

saved_papers = 0 
for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    pubmed_id = row["PMID"]

    if pubmed_id in not_found_ids:
        continue
    if os.path.exists(f"../../data/novelpy/PDF/pubmed_{pubmed_id}.pdf"):
        continue

        
    print(f"Processing {pubmed_id} ... ", end="", flush=True)

    doi = get_doi_for_pubmed(pubmed_id)

    if doi is None:
        with open("file_not_found.txt", "a") as f:
            print(pubmed_id, file=f)
        continue        
    
    try:
        doi2pdf.doi2pdf(doi, output=f"../../data/novelpy/PDF/pubmed_{pubmed_id}.pdf")
    except doi2pdf.main.NotFoundError:
        print("File not found, skipping.")
        with open("file_not_found.txt", "a") as f:
            print(pubmed_id, file=f)
        continue
    except KeyboardInterrupt:
        exit()
    except:
        print("Other error")
        with open("file_not_found.txt", "a") as f:
            print(pubmed_id, file=f)
        continue
        
    print("saved", flush=True)
    saved_papers += 1 
    time.sleep(10)

print(f"{saved_papers} saved.")
