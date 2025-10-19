import os
import sys 
import json
import pickle
import tqdm
import click
import pandas as pd
import logging 

from pydantic import BaseModel, Field
import instructor

from utils.config import create_settings

instructions="""
You are a scientific review assistant. For the abstract of scientific paper you are given, judge the novelty of the paper. Return the novelty score on the scale 0 to 9. The higher the score, the more novel the paper. 0 stands for non novel papers, 9 stands for highest novelty.
Output format: JSON {"score": <OUTPUT>} 
"""

class NoveltyScore(BaseModel):
    score: float = Field(ge=0.0, le=10.0, description="Novelty score.")

def eval_abstract(abstract, client):

    prompt = f"Judge the novelty of this abstract: '{abstract}'."

    result = client.chat.completions.create(
        response_model=NoveltyScore,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt}],
        temperature=0.1
    )
    print(result)
    return result.score

@click.command()
@click.argument("config_file")
@click.option("-m", "--model", default="jean-luc/tiger-gemma-9b-v3:fp16") 
def main(config_file, model):
    logger = logging.getLogger(__name__)

    settings = create_settings(config_file)
        
    client = instructor.from_provider(f"ollama/{model}")

    backup_file = f"{settings.output_prefix}_score_result_{model.replace('/', ':')}.pickle"
    
    if settings.input_file.endswith("feather"):
        df = pd.read_feather(settings.input_file)
    else:
        df = pd.read_csv(settings.input_file)
        
    ID = settings.id_column_name
    ABSTRACT = settings.abstract_column_name

    result = {}
    if os.path.exists(backup_file):
        with open(backup_file, "rb") as f:
            result = pickle.load(f)
        
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        if row[ID] in result:
            continue
        result[row[ID]] = eval_abstract(row[ABSTRACT], client)
        with open(backup_file, "wb") as f:
            pickle.dump(result, f)
        # with open(backup_file, "w") as f:
        #     json.dump(result, f)

    result = [{ID: key, "novelty_score": value} for key, value in result.items()] 
    pd.DataFrame(result).to_csv(f"../../results/{settings.output_prefix}_{model.replace('/', ':')}_result.csv")


if __name__ == "__main__":
    main()
