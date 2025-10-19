import tqdm 
import pandas as pd
import requests
import feedparser

def fetch_info_for_title(title):
    base_url = "http://export.arxiv.org/api/query"
    
    delete_chars = r",.(){}\\:-$[]%?^'\""
    for ch in delete_chars:
        title = title.replace(ch, " ")

    params = {
        "search_query": f"all:'{title}'",
        "start": 0,
        "max_results": 1,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"Error fetching data for title: {title}")
        raise ValueError("Failed to retrieve data from arXiv")

    feed = feedparser.parse(response.text)
    
    num_papers = len(feed.entries)
    if num_papers == 0:
        print(f"No papers found for title: {title}")
        raise ValueError("Empty feed.")

    entry = feed.entries[0]
    print(entry.keys())
    exit()
    return {
            "title": entry.title,
            "authors":  ";".join([author["name"] for author in entry.authors]),
            "id": entry.id,
            "published": entry.published,
            "summary": entry.summary
    }
           



df = pd.read_csv('arxiv_dataset_with_novelty.csv')
results = []

for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    title = row['title']
    new_row = fetch_info_for_title(title)
    new_row['novelty'] = row['novelty_score']
    results.append(new_row)

results_df = pd.DataFrame(results)
results_df.to_csv('arxiv_dataset_all_info.csv', index=False)
