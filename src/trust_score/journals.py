import logging 
import pickle
from pathlib import Path

import pandas as pd
import tqdm

from utils import send_request, eat_prefix

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s (%(module)s)] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_journal_abstracts(journal, year):
    # first retrieve journal ID
    params = {
        "filter": f"display_name.search:{journal}",
        "select": "id"
    }
    data = send_request(
        "https://api.openalex.org/journals",
        params,
        30
    )
    if data is not None:
        try:
         journal_id = eat_prefix(data['results'][0]['id'])
        except (IndexError, KeyError):
            logger.warning(f"Could not find journal ID for {journal}.")
            return []
    else:
        return []
    
    # now retrieve works for that journal and year
    work_list = []
    page = 1
    while page < 50:
        params = {
            "filter": f"primary_location.source.id:{journal_id},publication_year:{year-1}",
            "select": "id,cited_by_count",
            "per-page": 200,
            "page": page
        }
        data = send_request(
            "https://api.openalex.org/works",
            params,
            30
        )
        if data is None:
            break
        results = data['results']
        for work in results:
            if work['id'] is not None and work['cited_by_count'] is not None:
                work_list.append((work['id'], work['cited_by_count']))
        page += 1
    return work_list

df = pd.read_csv("journal_year.txt").drop_duplicates()
print(df)

results = {} 
result_filename = Path("tmp_journals.pickle")

if result_filename.exists():
    with open(result_filename, "rb") as f:
        results = pickle.load(f)

for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    journal = row['Journal']
    year = row['year']
    logger.info(f"{journal} ({year})")

    work_list = []
    works = get_journal_abstracts(journal, year)
    if len(works) == 0:
        logger.warning(f"No works found for {journal} ({year}).")
        results[(journal, year)] = None
        continue
    print(f"Found {len(works)} works.")
    # sort by citation count
    works = sorted(works, key=lambda x: x[1], reverse=True)
    results[(journal, year)] = works[:100]
    with open(result_filename, "wb") as f:
        pickle.dump(results, f)