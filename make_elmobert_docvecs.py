import os, csv, json, re
import os

import click
from tqdm import tqdm

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import *

#tf.logging.set_verbosity(tf.logging.ERROR)

from transformers import BertTokenizer, BertModel


def get_titles(df, column="title"):
    keywords = [preprocess_string(k, filters=[strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short]) for k in df[column].values]
    return keywords

def make_bert_docvecs(keywords, filename):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained("bert-base-uncased")

    docvecs = []
    for keyword in tqdm(keywords):
        doc = " ".join(keyword)
        encoded_input = tokenizer(doc, return_tensors='pt')
    
        output = bert(**encoded_input)
        docvec = output[0].mean(axis=1).squeeze().detach().numpy()
        docvecs.append(docvec)

    docvecs = np.array(docvecs)
    print("BERT -> docvecs.shape:", docvecs.shape)
    np.save(filename, docvecs)


def make_elmo_docvecs(keywords, filename):

    elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    init_op = tf.global_variables_initializer()
    
    docvecs = []
    for i in tqdm(range(0, len(keywords), 100)):
        docs = [" ".join(k) for k in keywords[i:i+100]]
        embeddings = elmo(
            docs,
            signature="default",
            as_dict=True)["elmo"]

        with tf.Session() as sess:
            sess.run(init_op)
            embeddings_np = sess.run(embeddings)

        docvec = embeddings_np.mean(axis=1)
        docvecs.append(docvec)

    docvecs = np.concatenate(docvecs)
    print("ELMo -> docvecs.shape:", docvecs.shape)
    np.save(filename, docvecs)

@click.command()
@click.argument('inputfile')
@click.argument('outputfile')
@click.option('-m', '--method', default="bert")
@click.option('--col', default="title")
def main(inputfile, outputfile, method, col):
    
    df = pd.read_csv(inputfile, index_col=0)
    df["abstract"] = df["abstract"].fillna("")

    keywords = get_titles(df, column=col)

    if method == "bert":
        make_bert_docvecs(keywords, filename=outputfile)
    elif method == "elmo":
        make_elmo_docvecs(keywords, filename=outputfile)
    else:
        print("Unknown method.")
    print("Finished.")
    
    

if __name__ == "__main__":
    main()
