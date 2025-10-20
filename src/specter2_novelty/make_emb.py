import sys 
import pandas as pd
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch
import tqdm

from utils.config import load_settings
from utils.utils import read_df

settings = load_settings()

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base', device_map="cuda")

#load base model
model = AutoAdapterModel.from_pretrained('allenai/specter2_base', device_map="cuda")

#load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
model.to("cuda")

df = read_df(settings.input_file)


ABSTRACT = settings.abstract_column_name 
for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    # # concatenate title and abstract
    if not row[ABSTRACT]:
        continue
    text_batch = [row['title'] + tokenizer.sep_token + row[ABSTRACT]]
    # # preprocess the input
    inputs = tokenizer(text_batch, padding=True, truncation=True,
                       return_tensors="pt", return_token_type_ids=False, max_length=512)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    output = model(**inputs)
    # # take the first token in the batch as the embedding
    embedding = output.last_hidden_state[:, 0, :]
    rowid = row[settings.id_column_name]
    if settings.output_prefix == "arxiv_dataset":
        rowid = rowid[len("http://arxiv.org/abs/"):]
        rowid = rowid.replace("/", ":")
        
    torch.save(embedding.detach().cpu(), f"../../data/emb/{settings.output_prefix}_{rowid}.pt")
