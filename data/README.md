# DATA SOURCES

This directory serves as a placeholder for datasets.

List here where to download data.

Petra's data/ directory is on [OwnCloud](https://owncloud.cesnet.cz/index.php/s/aidzr5vj2e1wtF0). 

+ arxiv_dataset 
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
