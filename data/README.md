# DATA SOURCES

This directory serves as a placeholder for datasets.

List here where to download data.

+ arxiv_dataset [petra_google_drive](https://drive.google.com/drive/folders/16yuxcArYI3Q6N1kpc_ZTGq1ejxLi3k_y?usp=sharing)
  - starting_point: https://www.kaggle.com/datasets/anasqaiser/research-papers-abstract-dataset
    + columns: title, summary, novelty   
  - created dataset: arxiv_dataset_all_info.csv  
    + columns: title, authors, id, published, summary, novelty
  - created dataset: directory with all PDF files, directory with TXT files (created from PDFs)
    + na google drive se to nevejde, kam s tim?
+ novelpy_sample
  - starting_point: `novelpy` package, `download_sample()`
  - created dataset: novelpy_data_sample.feather  
    + columns: PMID, title, abstract, year   
  - created dataset: novelpy_foster_result.feather
    + columns: PMID, score
