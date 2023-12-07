import os
import csv
import json
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

from collections import Counter, defaultdict
from pandas import DataFrame

from l2r import L2RRanker


def load_true_relevance(file: str) -> dict[int, list[tuple[int, float]]]:
    '''
    Helper function for Q2
    '''
    queries_to_judgements = defaultdict(list)
    with open(file) as f:
        reader = csv.reader(f)
        doc = next(reader)
        for doc in reader: 
            docid = int(doc[2])
            relevance = int(doc[-1])
            query = doc[0]
            queries_to_judgements[query].append((docid, relevance))

    return queries_to_judgements

def attribute_eval_pipeline(query: str, l2r: L2RRanker, doc_attributes: DataFrame, attributes:list[str], common_attributes:DataFrame) -> DataFrame:
    no_mmr = l2r.query(query)
    mmr_03 = l2r.query(query, mmr_lambda = 0.3)
    mmr_05 = l2r.query(query, mmr_lambda = 0.5)
    mmr_07 = l2r.query(query, mmr_lambda = 0.7)

    resps = [(no_mmr, 'No MMR'), (mmr_03, 'lambda = 0.3'), (mmr_05, 'lambda = 0.5'), (mmr_07, 'lambda = 0.7')]

    dfs = []
    for result, name in resps:
        dfs.append(make_df(result, name))

    attribute_df = pd.concat(dfs)
    attribute_df = attribute_df.merge(doc_attributes, on = 'docid')

    for attribute in attributes:
        plot_results(attribute_df, attribute, query, common_attributes)
        plt.show()

    return attribute_df


def plot_results(df: DataFrame, attribute: str, query: str, common_attributes:DataFrame):
    common_labels = common_attributes[common_attributes['attribute_label'] == attribute]['attribute_value'].to_list()
    df = df[df[attribute].isin(common_labels)]
    df = df[(df[attribute] != '') | (pd.isnull(df[attribute]) == False)]
    f = sns.barplot(data = df, x = attribute, y = 'rank', hue = 'ranker')
    f.set(xlabel = f'{attribute.title()}', ylabel = 'Average Rank')
    plt.xticks(rotation = 90)
    plt.title(f"Average Relevance Score for Query '{query.title()}' by {attribute.title()}")
    plt.legend(title = 'Ranker', loc = 'upper left', prop = {'size': 7})


def make_df(results, name:str) -> DataFrame:
    df = pd.DataFrame(results)
    df.columns = ['docid', 'score']
    df['rank'] = df['score'].rank(method = 'dense', ascending = False)
    df['ranker'] = name
    return df


def get_docid_to_categories(documents_path:str, categories_key:str = 'categories', save_path:str = '') -> dict[int, list[str]]:
    docid_to_categories = {}
    with open(documents_path) as f: 
        doc = f.readline()
        while doc: 
            doc = json.loads(doc)
            docid = doc['docid']
            categories = doc[categories_key]
            docid_to_categories[docid] = categories 
            doc = f.readline()
        
    if save_path != '' and not os.path.isfile(save_path):
        with open(save_path) as f: 
            json.dump(docid_to_categories, f)
    return docid_to_categories


def get_docid_to_network_features(documents_path:str, save_path:str='') -> dict[int, dict[str, float]]:
    docid_to_network_features = {}
    with open(documents_path) as f: 
        doc = f.readline()
        while doc: 
            docid, pagerank, authority_score, hub_score = doc[0], doc[1], doc[2], doc[3]
            features = {}
            features['pagerank'] = pagerank 
            features['authority_score'] = authority_score
            features['hub_score'] = hub_score
            docid_to_network_features[docid] = features 
            doc = f.readline()
    if save_path != '':
        with open(save_path) as f: 
            json.dump(docid_to_network_features, f)
    return docid_to_network_features


def load_person_attributes(file: str, outfile: str) -> dict[int, list]:
    '''
    Helper function for question 3

    File header order: 
    title, ethnicity, gender, religious_affilitation, political_party, docid
    '''
    docs_to_attributes = {}

    ethnicity_counts = Counter()
    gender_counts = Counter()
    religious_affilitation_counts = Counter()
    political_party_counts = Counter()

    with open(file, 'r') as f: 
        reader = csv.reader(f)
        doc = next(reader)
        for doc in reader: 

            docid = int(doc[5])
            docs_to_attributes[docid] = doc[:-1]

            # accumulate most common labels 
            ethnicity = doc[1]
            gender = doc[2]
            religious_affilitation = doc[3]
            political_party = doc[4]

            # avoid counting missing vals
            if ethnicity != '': ethnicity_counts[ethnicity] += 1 
            if gender != '': gender_counts[gender] += 1 
            if religious_affilitation != '': religious_affilitation_counts[religious_affilitation] += 1 
            if political_party != '': political_party_counts[political_party] += 1 

    iterator = [('ethnicity', ethnicity_counts.most_common(10)), 
                ('gender', gender_counts.most_common(10)), 
                ('religion', religious_affilitation_counts.most_common(10)), 
                ('politics', political_party_counts.most_common(10))
                ]
    
    if not os.path.isfile(outfile):
        with open(outfile, 'a') as f: 
            writer = csv.writer(f)
            writer.writerow(['attribute_label', 'attribute_value', 'n_occurences'])
            for attribute_label, l in iterator:
                for tup in l:
                    attribute_value, n_occurences = tup 
                    writer.writerow([attribute_label, attribute_value, n_occurences])

    return docs_to_attributes
