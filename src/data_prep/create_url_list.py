import os
import tqdm
from utils.utils import read_df
from alex_utils import get_url_for_pubmed

    

df = read_df("../../data/cell/cell_papers_selected_novelty.parquet")


for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    pubmed_id = row["pubmed_id"]
    if os.path.exists(f"../../data/cell/PDF/pubmed_{pubmed_id}.pdf"):
        continue
    pdf_url = get_url_for_pubmed(pubmed_id)

    assert pdf_url is not None
    
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
