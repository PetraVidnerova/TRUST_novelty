import logging 
import pickle 
from pathlib import Path
import requests

import pandas as pd
import tqdm 
import torch
import torch.nn.functional as F
from tenacity import retry, stop_after_attempt, retry_if_exception_type, before_sleep_log

from utils import download_abstract, get_related_works, create_abstract, eat_prefix
from embeddings import Embeddings

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s (%(module)s)] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

@retry(
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(requests.exceptions.HTTPError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    retry_error_callback=lambda _: None,
)
def send_request(url, params, timeout):
    response = requests.get(
        url,
        params=params,
        timeout=timeout
    )
    response.raise_for_status()
    data = response.json()
    return data


class Evaluator():

    def __init__(self, datadir):
        # self load saved abstracts
        #self.abstracts = self._load_abstracts(datadir) 
        self.emb_model = Embeddings()
        
    def _load_abstracts(self, datadir):
        filename = Path(datadir) / "abstracts.parquet"
        if filename.exists():
            df = pd.read_parquet(filename)[["PaperProjectID", "abstract"]].dropna()
            result = (
                df
                .set_index("PaperProjectID")["abstract"]
                .to_dict()
            )
            return result 
        else:
            # return emtpy directory 
            return dict()


        
    def _get_data_for_target(self, alexid, include_abstract=True):
        """
        Fetch only abstract and related works metadata for a given OpenAlex ID.
        """
        OPENALEX_WORK_URL = "https://api.openalex.org/works/{}"

        alexid = eat_prefix(alexid)

        # Request ONLY the needed fields
        if include_abstract:
            params = {
                "select": "abstract_inverted_index,related_works"
            }
        else:
            params = {
                "select": "related_works"
            }

            
        data = send_request(
            OPENALEX_WORK_URL.format(alexid),
            params,
            30
        )

        if data is None:
            return None
        
        related_works = data.get("related_works", [])
        result = {
            "related_works": related_works
        }
        
        if include_abstract:
            abstract = create_abstract(
                data.get("abstract_inverted_index", None)
            )
            result["abstract"] = abstract 

        
        return result

    def _get_data_for_related(self, works):
        """ Fetch title and abstract for a list of OpenAlex IDs. """
        OPENALEX_WORK_URL = "https://api.openalex.org/works"

        assert len(works) < 200 
        works = list(map(eat_prefix, works))
        
        filter_ids = "|".join(works)
        if self.titles_only:
            selection = "id,title"
        else:
            selection = "id,title,abstract_inverted_index"
        params = {
            "filter": f"ids.openalex:{filter_ids}",
            "select": selection 
        }
    

        data = send_request(
            OPENALEX_WORK_URL,
            params,
            30
        )
        if data is None:
            return None
        
        results = {}
        for work in data.get("results", []):
            results[work["id"]] = {
                "title": work.get("title", None),
                "abstract": create_abstract(
                    work.get("abstract_inverted_index", None)
                )
            }
        return results 
        
    # def _get_abstract(self, pid, alexid, title):
    #     if pid in self.abstracts:
    #         return title, self.abstracts[pid]
    #     else:
    #         target_title, target_abstract = download_abstract(alexid)
    #         if not title:
    #             title = target_title
    #         if target_abstract is None:
    #             # replace abstract by title 
    #             logger.warning(f"Missing abstract for {pid}.")
    #             return title, title
    #         else:
    #             return title, target_abstract

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor):
        return F.cosine_similarity(a.flatten(), b.flatten(), dim=0)

    def overall_similarity(self, target, related):
        result = 0
        target = target[0] 
        for rel in related:
            result += self._cosine_similarity(target, rel)
        result /= related.shape[0]
        return result
    
    def eval(self, pid, doi, alexid, title, abstract=None):
        target_data = self._get_data_for_target(alexid,
                                                include_abstract=abstract is None)

        if target_data is None:
            return None
        target_title = title
        target_abstract = abstract if abstract is not None else target_data.get("abstract", None)
        if not isinstance(target_abstract, str):
            target_abstract = None
        related_works = target_data.get("related_works", None)

        if related_works is None:
            # todo: find related works manualy based on topics
            return None

        if target_abstract is None:
            self.titles_only = True
        else:
            self.titles_only = False


        related_data = self._get_data_for_related(related_works)
        # extract titles and abstract
        # if title does not exists, we ignore the item
        titles = [item["title"] for item in related_data.values() if item["title"] is not None]
        if not self.titles_only:
            abstracts = [item["abstract"] for item in related_data.values() if item["title"] is not None]
            n_abstracts = len([a for a in abstracts if a is not None])

            # we use abstracts only if we have at least 5
            if n_abstracts < 5:
                self.titles_only = True
            else:
                #keep only those tuples where we have both of them (abstract is not None) 
                related_titles_abstracts = zip(titles, abstracts)
                related_titles_abstracts = [(t, a) for (t, a) in related_titles_abstracts if a is not None and isinstance(a, str)]

        if self.titles_only:
            related_titles_abstracts = [(t, None) for t in titles]

                
        target_emb = self.emb_model.embed([(target_title, target_abstract)], titles_only=self.titles_only)
        related_embs = self.emb_model.embed(related_titles_abstracts, titles_only=self.titles_only)
        result = self.overall_similarity(target_emb, related_embs)
        return {
            "score": 1.0 - result.item(),
            "n_related": len(related_titles_abstracts),
            "titles_only": self.titles_only,
            "message": "success" 
        }
            
def main():
    logger.setLevel("DEBUG")


    evaluator = Evaluator("../../data/soutez/")
    
    DOI = 'DOI (as URL)'
    OPENALEXID = 'OpenAlexID (as URL)'
    
    #df = pd.read_csv("../../data/soutez/Metadata file COMBINED.csv")
    TASK = "cell"
    df = pd.read_csv(f"../../data/soutez/{TASK}.csv")

    result_filename = Path(f"tmp_result_{TASK}.pkl")
    if result_filename.exists():
        with open(result_filename, "rb") as f:
            main_result = pickle.load(f)
        logger.info(f"Loaded existing results for {len(main_result)} items.")
    else:   
        logger.info("No existing results found, starting from the beginning.")
        main_result = {} 
    
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        pid = row["PaperProjectID"]

        if pid in main_result:
            continue

        doi = row[DOI]
        alexid = row[OPENALEXID]
        title = row["Title"]
        abstract = row.get("Abstract", None)
        
        if not title:
            result = None
        else:    
            result = evaluator.eval(pid, doi, alexid, title, abstract)

        if result is None:
            result = {"score": -1.0, "titles_only": None, "message": "no_data"}
        main_result[pid] = result
        print(pid, result["score"], result["titles_only"], result["message"])

        if i % 10 == 0:
            with open(f"tmp_result_{TASK}.pkl", "wb") as f:
                pickle.dump(main_result, f)


if __name__ == "__main__":
    main()