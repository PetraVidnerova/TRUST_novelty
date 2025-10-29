import os
import pickle
import tqdm
import click
import pandas as pd
import logging 

from ollama import Client

from utils.config import create_settings
from utils.utils import read_df

instructions="""
You are a scientific review assistant. For the abstract of scientific paper you are given, write a short review judging the novelty of the paper.
"""


def eval_abstract(text, client, model):

    prompt = f"Judge the novelty of this abstract: '{text}'."
    
    custom_options = {"temperature": 0.1}

    response = client.chat(model=model, messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt}],
    options=custom_options)

    res_text = response["message"]["content"]
    print(res_text)
    return res_text

def load_full_text(task_name, paper_id):
    paper_id = "arxiv_" + paper_id[len("http://arxiv.org/abs/"):]
    try: 
        with open(f"../../data/{task_name}/TXT/{paper_id}.txt") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"../../data/{task_name}/TXT/{paper_id}.txt")
        return None
    return text  
    
@click.command()
@click.argument("config_file")
@click.option("-m", "--model", default="jean-luc/tiger-gemma-9b-v3:fp16") 
def main(config_file, model):
    logger = logging.getLogger(__name__)

    settings = create_settings(config_file)
        
    #client = instructor.from_provider(f"ollama/{model}")

    client = Client(host='http://localhost:11434')


    backup_file = f"{settings.output_prefix}_summary_{model.replace('/', ':')}.pickle"
    
    df = read_df(settings.input_file)
        
    ID = settings.id_column_name
    ABSTRACT = settings.abstract_column_name

    result = {}
    if os.path.exists(backup_file):
        with open(backup_file, "rb") as f:
            result = pickle.load(f)
        
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        if row[ID] in result:
            continue
        text = load_full_text(settings.task_name, row[ID])
        if text is None:
            result[row[ID]] = None
        else:
            result[row[ID]] = eval_abstract(text, client, model)
        with open(backup_file, "wb") as f:
            pickle.dump(result, f)
        # with open(backup_file, "w") as f:
        #     json.dump(result, f)

    result = [{ID: key, "summary": value} for key, value in result.items()] 
    pd.DataFrame(result).to_csv(f"../../results/{settings.output_prefix}_{model.replace('/', ':')}_summary.csv")


if __name__ == "__main__":
    main()
