import os
import nltk
from numpy.core.fromnumeric import reshape
import spacy
from nltk import word_tokenize, sent_tokenize
import neuralcoref
from transformers import BertTokenizer, BertModel as BM
import torch
import re
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import pickle


from collections import defaultdict, Counter

from transformers.models import bert
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

from scipy.spatial.distance import cosine

labels = ['Judiciary',
    'Elected Central Govt',
    'Govt. Bureaucrat',
    'State Govt controlled by ruling party in centre',
    'State Govt controlled by oppositions',
    'Political Party (ruling in centre)',
    'Political Party in opposition',
    'Citizen and Activists',
    'International Figures',
    'News Editors',
    'Farmers',
    'NA'
]

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

def load_prelabeled_entities():
    ent_labels = dict()

    with open('known_ents.txt', 'r+') as f: 
        for line in f.readlines():
            ls = list(line.split())
            label = int(ls[-1])
            ls.pop()
            ent_labels[' '.join(ls)] = label
    return ent_labels
class BertModel:
    
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BM.from_pretrained('bert-base-uncased',output_hidden_states=True)
        
    def get_BERT_embeddings(self,sentence):  # returns array of size 1xtokens_sizex768
        model_input = self.tokenizer(sentence, return_tensors='pt', padding = True, truncation=True, add_special_tokens = True)
        model_input.requires_grad = False
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        output = self.model(**model_input)
        embeddings = torch.squeeze(output[2][-1])
        cls_embed = torch.squeeze(torch.index_select(embeddings, 0, torch.tensor([0])))
        return cls_embed.detach().clone().numpy(), embeddings.detach().clone().numpy()
    
    def get_mapping(self, sentence):

        tokenized_sentence_aux = ['[CLS]']
        word_to_token_mapping = []
        for word in word_tokenize(sentence):
            word_to_token_mapping.append(len(tokenized_sentence_aux))    
            tokenized_sentence_aux.extend(self.tokenizer.tokenize(word))
        
        return word_tokenize(sentence), word_to_token_mapping, tokenized_sentence_aux

def get_entity_dict_kmeans(article):
    bertmodel = BertModel()
    docs = load_dataset()

    article = docs[1]


    print(article)
    entities_text = []
    X = []
    for sentence in sent_tokenize(article):
        doc = nlp(sentence)
        entities =[X for X in doc.ents if X.label_ in ['ORG', 'PERSON'] ]
        cls_embed, embeddings = bertmodel.get_BERT_embeddings(sentence)
        
        sentence_words, word_to_token_mapping, bert_tokenized = bertmodel.get_mapping(sentence)

        for ent in entities:
            words = word_tokenize(sentence[ent.start_char: ent.end_char])
            start_ind = sentence_words.index(words[0])
            end_ind = sentence_words.index(words[-1])
            
            # print(bert_tokenized[word_to_token_mapping[start_ind] : word_to_token_mapping[end_ind + 1]])
            entitiy_embedding = embeddings[word_to_token_mapping[start_ind] : word_to_token_mapping[end_ind + 1]]
            # print(word_to_token_mapping[start_ind] , word_to_token_mapping[end_ind + 1], ent.text)
            entitiy_embedding = np.concatenate([entitiy_embedding.mean(axis = 0).reshape(-1), cls_embed])

            X.append(entitiy_embedding)
            # print(X[-1].shape)
            entities_text.append(ent.text)
    
    X = np.array(X)
    print(X.shape)
    
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    clusters = defaultdict(list)
    
    for ent, label in zip(entities_text, kmeans.labels_):
        clusters[label].append(ent)
    
    print(clusters)

def get_entity_dict_WDcoref(article):
    
    article  = nlp(article)._.coref_resolved
    print(article)
    entities_text = []
    X = []

    ent_embedding_dict = defaultdict(list)


    for sentence in sent_tokenize(article):
        doc = nlp(sentence)
        entities =[X for X in doc.ents if X.label_ in ['ORG', 'PERSON'] ]
        cls_embed, embeddings = bertmodel.get_BERT_embeddings(sentence)
        
        sentence_words, word_to_token_mapping, bert_tokenized = bertmodel.get_mapping(sentence)
        lastchar = 0

        

        sentence_words = [word.strip("'") for word in sentence_words]
        for ent in entities:
            try : 
                words = word_tokenize(sentence[ent.start_char: ent.end_char])
                words = [word.strip("'") for word in words if len(word.strip("'")) > 0]
                
                start_ind = sentence_words.index(words[0], lastchar)
                end_ind = sentence_words.index(words[-1], start_ind)

                lastchar = end_ind
                
                # print(bert_tokenized[word_to_token_mapping[start_ind] : word_to_token_mapping[end_ind + 1]])
                entitiy_embedding = embeddings[word_to_token_mapping[start_ind] : word_to_token_mapping[end_ind + 1]]

                if(entitiy_embedding.shape[0] == 0):
                    print(ent.text, words, lastchar, start_ind,  end_ind)
                # print(word_to_token_mapping[start_ind] , word_to_token_mapping[end_ind + 1], ent.text)
                entitiy_embedding = np.concatenate([entitiy_embedding.mean(axis = 0).reshape(-1), cls_embed])

                # X.append(entitiy_embedding)
                # print(X[-1].shape)
                # entities_text.append(ent.text)

                ent_embedding_dict[ent.text].append(entitiy_embedding)
            except : 
                pass        
    ent_embedding_dict = dict(ent_embedding_dict)
    for k, v in ent_embedding_dict.items():
        ent_embedding_dict[k] = np.array(v).mean(axis = 0)
        

    return  ent_embedding_dict


def print_similarity_matrix(entity_doc1, entity_doc2):

    sims = Counter()
    for ent1, emb1 in entity_doc1.items():
        for ent2, emb2 in entity_doc2.items():
            sims[(ent1, ent2)] = 1 - cosine(emb1, emb2) 
    
    print("10 most similar entities : ", sims.most_common(10))
    print("10 least similar entities : ", sims.most_common()[-10:])



def incremental_candidate_composition(X, orig_clusters, singletons, doc_ids, all_entitities):
    cur_clusters_no = len(orig_clusters)

    clusters_mean_embedding = dict()

    cluster_index = 0
    cluster_sizes = dict()
    clusters = dict()
    threshold = 0.3
    for link, ids in orig_clusters.items():
        ids_embeddings = [X[id] for id in ids]
        clusters_mean_embedding[cluster_index] = np.array(ids_embeddings).mean(axis = 0)
        cluster_sizes[cluster_index] = len(ids)
        clusters[cluster_index] = ids
        cluster_index += 1


    for singleton_id in singletons : 

        # (e1, d1), (e3, d3)..  -> mean - > cluster embedding 
        similarity_scores = [1 - cosine(X[singleton_id], clusters_mean_embedding[cluster_id]) for cluster_id in range(cluster_index)]
        max_cluster = np.argmax(similarity_scores)
        
        if(similarity_scores[max_cluster] < threshold):
            clusters_mean_embedding[cluster_index] = X[singleton_id]
            clusters[cluster_index] = [singleton_id]
            cluster_sizes[cluster_index] = 1
            cluster_index += 1
            print("forming new cluster")
        else:
            print("adding entitity ", all_entities[singleton_id] , " to cluster ", max_cluster)
            sz = cluster_sizes[max_cluster]
            clusters_mean_embedding[max_cluster] = clusters_mean_embedding[max_cluster] * ((float)(sz) / (sz + 1))  + X[singleton_id]/(sz + 1)
            clusters[max_cluster].append(singleton_id)
            cluster_sizes[max_cluster] += 1

    ent_labels = load_prelabeled_entities()
    for index, cluster in clusters.items():
        if(all_entities[cluster[0]] in list(ent_labels.keys())):
            tag = labels[ent_labels[all_entities[cluster[0]]]]
        else:
            tag = 'NA'
        print("Printing cluster no : ", index, ' with tag : ', tag)
        
        ents = [all_entities[ent_id] for ent_id in cluster]

        print(ents)

            



if __name__ == '__main__':

    bertmodel = BertModel()
    print(load_prelabeled_entities())
    docs = load_dataset()

    dbpedia_links = pickle.load(open('dbpedia_links_dict.pkl', 'rb'))
    doc_ids = []
    all_entities = []
    X = []
    for i in range(50):
        article = docs[i]
        ents = get_entity_dict_WDcoref(article)

        for ent, emb in ents.items():
            
            doc_ids.append(i)
            all_entities.append(ent)
            X.append(emb)
        
    
    X = np.array(X)
    clustering = AgglomerativeClustering(n_clusters=None,distance_threshold=20).fit(X)
    sets = list(clustering.labels_)

    orig_clusters = defaultdict(list)
    singletons = []

    for i, ent in enumerate(all_entities):
        if(ent in dbpedia_links):
            orig_clusters[dbpedia_links[ent]].append(i)
        else : 
            singletons.append(i)


    print(orig_clusters)

    # known_ents = []
    # for k, v in orig_clusters.items():
    #     known_ents.extend([all_entities[v1] for v1 in v])
    

    # with open("known_ents.txt", "w+") as f:
    #     for v1 in known_ents:
    #         f.write(str(v1) + '\n')
    # f.close()
    # print(len(singletons)/len(all_entities))    

    incremental_clusters = incremental_candidate_composition(X, orig_clusters, singletons, doc_ids, all_entities)
    # print(sets, all_entities, doc_ids)



    # article1 = docs[48]
    # article2 = docs[67]

    # print_similarity_matrix(get_entity_dict_WDcoref(article1), get_entity_dict_WDcoref(article2))

    # print(docs[0])

    
    



    # print(sample_doc._.coref_clusters)


