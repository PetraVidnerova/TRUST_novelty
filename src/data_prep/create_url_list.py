import os
import tqdm
from utils.utils import read_df
from alex_utils import get_url_for_pubmed

    
""" 
df = read_df("../../data/cell/cell_papers_selected_novelty.parquet")
# with open("missing_dois.txt", "r") as f:
#     for line in f:
#         pubmed_id = int(line.strip()) 
#         pdf_url = get_url_for_pubmed(pubmed_id)
#         if pdf_url is None:
#             print(f"Skipping {pubmed_id}")
#             continue
#         with open("download_papers_missingdois.html", "a") as f:
#             print(
#                 f"<a href='{pdf_url}' download='pubmed_{pubmed_id}.pdf'> pubmed_{pubmed_id}.pdf </a><br>",
#                 file=f
#             )
# exit()
        
for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    doi = row["doi"]
    # if doi is not None:
    #     continue
    pubmed_id = row["pubmed_id"]
    if os.path.exists(f"../../data/cell/PDF/pubmed_{pubmed_id}.pdf"):
        continue
    pdf_url = get_url_for_pubmed(pubmed_id)

    if pdf_url is None:
        with open("pdf_url_not_found.txt", "a") as f:
            print(pubmed_id, file=f)
        continue
    
    with open("download_papers.html", "a") as f:
        print(
            f"<a href='{pdf_url}' download='pubmed_{pubmed_id}.pdf'> pubmed_{pubmed_id}.pdf </a><br>",
            file=f
        )
 """

with open("pdf_url_not_found.txt", "r") as f:
    for line in f:
        pubmed_id = line.strip()
        if os.path.exists(f"../../data/cell/PDF/pubmed_{pubmed_id}.pdf"):
            continue
        pdf_url = get_url_for_pubmed(pubmed_id)

        if pdf_url is None:
            print(f"Still not found: {pubmed_id}")
            continue
        else:
            print(f"Found now: {pubmed_id}")
        
        with open("download_papers.html", "a") as f:
            print(
                f"<a href='{pdf_url}' download='pubmed_{pubmed_id}.pdf'> pubmed_{pubmed_id}.pdf </a><br>",
                file=f
            )
    
    

    # doi = row["doi"]
    # if doi in not_found_ids:
    #     continue
    # print(f"Processing {doi} ... ", end="", flush=True)
    # try:
    #     doi2pdf.doi2pdf(doi, output=f"../../data/cell/PDF/pubmed_{pubmed_id}.pdf")
    # except doi2pdf.main.NotFoundError:
    #     print("File not found, skipping.")
    #     with open("file_not_found.txt", "a") as f:
    #         print(doi, file=f)
    #     continue
    # except:
    #     print("Other error")
    #     with open("file_not_found.txt", "a") as f:
    #         print(doi, file=f)
    #     continue
        
    # print("saved", flush=True)
    # time.sleep(10)
