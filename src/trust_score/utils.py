import logging 
import requests
import time

import pyalex
from tenacity import (
    retry, stop_after_attempt, retry_if_exception_type, 
    before_sleep_log, wait_random_exponential
)

logger = logging.getLogger("__main__")

@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type(requests.exceptions.HTTPError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    retry_error_callback=lambda _: None
)
def send_request(url, params, timeout):
    params["mailto"] = "petra@cs.cas.cz"
    response = requests.get(
        url,
        params=params,
        timeout=timeout
    )
    if response.status_code == 404: 
        logger.warning(f"Data not found at {url}.")
        return None
    response.raise_for_status()
    data = response.json()
    return data



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
