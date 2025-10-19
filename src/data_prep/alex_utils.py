import pyalex

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
