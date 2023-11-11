from openie import StanfordOpenIE
import os
import nltk
import spacy
# import spacy_dbpedia_spotlight
from nltk import word_tokenize, sent_tokenize
import neuralcoref
import transformers

import pandas as pd
    
import re
# nlp = spacy.load('en')
nlp = spacy.load('en_core_web_sm')
# nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)

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
        if not filename.endswith(".txt"):
            continue
        p = re.sub("[^a-zA-Z]+", '', p)
        s[p] = s.get(p, 0) + 1
        with open("farmer_law_data/" + filename, 'rb') as f:

            docs.append(f.read().decode('utf-8'))
    
    docs = [remove_urls(doc) for doc in docs]

    docs = [remove_html(doc) for doc in docs]

    
    return docs



if __name__ == '__main__':
    docs = load_dataset()
    # print(len(docs))

    properties = {
            'openie.affinity_probability_cap': 2 / 3,
        }

    with StanfordOpenIE() as client:
        for doc in docs[:10]:

            print("**********************************Printing for doc*********************************** ")
            doc  = nlp(doc)._.coref_resolved
            doc = nlp(doc)
            entities =[X.text for X in doc.ents if X.label_ in ['ORG', 'PERSON'] ]
            # print(entities)
            text = str(doc)
            # print('Text: %s.' % text)
            triples = [dict(triple) for triple in client.annotate(text)]
            useful_relations = []
            for triple in triples:
                isUseful = False
                subj = nlp(triple['subject'])
                obj = nlp(triple['object'])

                subj = [X.label_ for X in subj.ents]
                obj = [X.label_ for X in obj.ents]

                isUseful = ('ORG' in subj) or ('PER' in subj)

                isUseful = isUseful and (('ORG' in obj) or ('PER' in obj))

                # isUseful = True
                if(isUseful):
                    useful_relations.append(triple)

            for triple in useful_relations:
                print('|-', triple)
            

    # for doc in docs[:10]:
        
        

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

