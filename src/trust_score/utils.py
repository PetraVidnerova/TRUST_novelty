import logging 
import pyalex
import requests
import time

logger = logging.getLogger("__main__")


def create_abstract(abstract_index):
    if abstract_index is None:
        return None
    maximum = 0
    for indexes in abstract_index.values():
        m = max(indexes)
        if m > maximum:
            maximum = m
    words = [""] * (maximum+1)
    for w, indexes in abstract_index.items():
        for i in indexes:
            words[i] = w
    return " ".join(words)

def download_abstract(alexid):
    for trial in range(5):
        try:
            work = pyalex.Works()[alexid]
            title = work['title']
        
            if work['abstract']:
                return title, work['abstract']
            elif work['abstract_inverted_index']:
                abstract = work['abstract_inverted_index']
                return title, create_abstract(abstract)
            else:
                logger.warning(f"No abstract in PyAlex for {alexid}.")
                return title, None 
        except requests.exceptions.HTTPError:
            logger.warning(f"HTTPError for PyAlex ... trial {trial}, going to sleep for 10s.")
            time.sleep(10)
    return None, None
        

def get_related_works(alexid):
    try:
        work = pyalex.Works()[alexid]
   
        related = work['related_works']
        if not related:
            logger.warning(f"Missing related works for {alexid}.")
            return None

        return [r[len("https://openalex.org/"):] for r in related] 
    except requests.exceptions.HTTPError:
        logger.warning(f"HTTPError for PyAlex")
        return None

def eat_prefix(alexid):
    PREFIX = "https://openalex.org/"
    if alexid.startswith(PREFIX):
        return alexid[len(PREFIX):]
    else:
        return alexid
