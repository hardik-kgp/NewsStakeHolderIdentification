import os
import nltk
import spacy
# import spacy_dbpedia_spotlight
from nltk import word_tokenize, sent_tokenize
# import neuralcoref
import transformers

import pandas as pd

import re
# nlp = spacy.load('en')
# nlp = spacy.load('en_core_web_lg')
# # neuralcoref.add_to_pipe(nlp)

# nlp.add_pipe('dbpedia_spotlight')

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
    
def load_dataset():
    all_files = os.listdir("farmer_law_data/")
    docs = []
    s = dict()
    for filename in all_files:
        p = filename
        p = re.sub("[^a-zA-Z]+", '', p)
        s[p] = s.get(p, 0) + 1
        with open("farmer_law_data/" + filename, 'rb') as f:

            docs.append(f.read().decode('utf-8'))
    
    docs = [remove_urls(doc) for doc in docs]

    docs = [remove_html(doc) for doc in docs]

    print(len(s))
    print(s)
    return docs



if __name__ == '__main__':
    docs = load_dataset()
    print(len(docs))
    # sample_doc = nlp(docs[1])
    # print(docs[1])
    # print([(X.start, X.end, X.text) for X in sample_doc.ents])
    
    # clusters = {}
    # for X in sample_doc.ents:
    #     if(X.kb_id_ not in clusters):
    #         clusters[X.kb_id_] = []
    #     clusters[X.kb_id_].append(X.text)

    # print(clusters)
    # print(sample_doc._.coref_clusters)

    # print(type(sample_doc), type(sample_doc._.coref_resolved), sample_doc._.coref_resolved)