# DATA SOURCES

This directory serves as a placeholder for datasets.

List here where to download data.

+ arxiv_dataset
  - starting_point: https://www.kaggle.com/datasets/anasqaiser/research-papers-abstract-dataset
    + columns: title, summary, novelty   
  - crated dataset: arxiv_dataset_all_info.csv  
    + columns: title, authors, id, published, summary, novelty
+ novelpy_sample
  - starting_point: `novelpy` package, `download_sample()`
  - created dateset: novelpy_data_sample.feather  
    + columns: PMID, title, abstract, year   
  - created dataset: novelpy_foster_result.feather
    + columns: PMID, score
