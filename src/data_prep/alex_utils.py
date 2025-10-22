import pyalex
import requests

pyalex.config.email = "petra@cs.cas.cz"

pyalex.config.max_retries = 2
pyalex.config.retry_backoff_factor = 0.1
pyalex.config.retry_http_codes = [429, 500, 503]

def clean_title(title):
    title = title.replace(",", "").replace(".", "").replace("\n", " ")
    title = title.replace("-", "").replace("!", "")
    title = title.replace("&amp;", "").replace("&", "")
    title = " ".join(title.split())
    return title.lower()

def get_openalex_id_from_arxiv(arxiv_url):

    number = arxiv_url[len("http://arxiv.org/abs/"):]
    if "v" in number:
        number = number[:number.index("v")]
    for query in f"http://arxiv.org/abs/{number}", f"https://arxiv.org/abs/{number}":

        ws = pyalex.Works().filter(**{  # "primary_location.source.id": ID,
            "primary_location.landing_page_url": query
        }).get()

        if len(ws) > 0:
            return [w['id'] for w in ws]  
    return None

def get_openalex_id_from_title(title):
    collection = pyalex.Works().search_filter(title=clean_title(title)).get()
    results = []
    for paper in collection:
        if clean_title(paper["title"]) != clean_title(title):
            continue
        
        results.append(paper["id"])
    return results

def get_url_for_doi(doi):
    res = pyalex.Works()["https://doi.org/" + doi]
    try:
        return res["primary_location"]["pdf_url"]
    except KeyError:
        return None

def get_url_for_pubmed(pubmed):

    url = f"https://api.openalex.org/works/pmid:{pubmed}"

    trials = 0
    while trials < 10:
        trials += 1
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            try:
                if data["primary_location"]["pdf_url"] is None:
                    
                    for location in data["locations"]:
                        if location["pdf_url"] is not None:
                            print("Found alternative location for", pubmed)
                            return location["pdf_url"]
                    return None
                else:
                    return data["primary_location"]["pdf_url"]
            except KeyError:
                return None
            
                

