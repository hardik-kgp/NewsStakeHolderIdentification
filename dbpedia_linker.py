import os
import nltk
from numpy.core.fromnumeric import reshape
import spacy
from nltk import word_tokenize, sent_tokenize

import re
import pickle
import spacy_dbpedia_spotlight
# load your model as usual
nlp = spacy.load('en_core_web_lg')
# add the pipeline stage
nlp.add_pipe('dbpedia_spotlight')
# get the document


num_clusters = 5

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
    
def load_dataset():
    all_files = os.listdir("farmer_law_data/")
    docs = []
    for filename in all_files:
        with open("farmer_law_data/" + filename, 'rb') as f:
            docs.append(f.read().decode('utf-8'))
    
    docs = [remove_urls(doc) for doc in docs]

    docs = [remove_html(doc) for doc in docs]
    return docs


if __name__ == '__main__':
    docs = load_dataset()
    dbpedia_links = dict()
    
    for article in docs : 
        try : 
            doc = nlp(article)
            # print(doc.ents)
            # see the entities
            entities =[X for X in doc.ents ]
            # print(entities)
            # inspect the raw data from DBpedia spotlight
            for ent in entities:
                try : 
                    dbpedia_links[ent.text] = str(ent.kb_id_)
                except : 
                    print(ent)
            print("done !")
        except : 
            print(article)
    pickle.dump(dbpedia_links, open('dbpedia_links_dict.pkl','wb'))
    
        