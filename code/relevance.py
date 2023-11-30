import csv 
import numpy as np 
from tqdm import tqdm
import pandas as pd
import csv

def map_score(search_result_relevances: list[int], cut_off=10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_results: A list of 0/1 values for whether each search result returned by your 
                        ranking function is relevant
        cut_off: The search result rank to stop calculating MAP. The default cut-off is 10;
                 calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    precision = []
    cut_off = min(len(search_result_relevances), cut_off)
    for i in range(len(search_result_relevances[:cut_off])):
        part = search_result_relevances[:(i + 1)]
        if part[i] == 0:
            precision.append(0)
            continue
        num_pos = part.count(1.0)
        precision.append(num_pos / (i + 1))

    if np.sum(precision) == 0:
        return 0
    map = np.sum(precision) / cut_off
    return map

def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_off=10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: 
            A list of relevance scores for the results returned by your ranking function in the
            order in which they were returned. These are the human-derived document relevance scores,
            *not* the model generated scores.
            
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score in descending order.
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    actual_rating = []
    ideal_rating = []
    cut_off = min(len(ideal_relevance_score_ordering), len(search_result_relevances), cut_off)
    for i in range(cut_off):
        rating = search_result_relevances[i]
        ideal_val = ideal_relevance_score_ordering[i]
        if i == 0:
            actual_rating.append(rating)
            ideal_rating.append(ideal_val)
        elif i > 0:
            rating = rating / np.log2(i + 1)
            ideal_val = ideal_val / np.log2(i + 1)
            actual_rating.append(rating)
            ideal_rating.append(ideal_val)

    if np.sum(ideal_rating) > 0:
        return np.sum(actual_rating) / np.sum(ideal_rating)
    else:
        return 0


def map_queries_to_judgements(relevance_data_filename:str) -> dict[str, list[tuple[int]]]:
    with open(relevance_data_filename) as f:
            reader = csv.reader(f)
            queries_to_judgements = {}
            header = next(reader)
            query_idx = header.index('query')
            rel_idx = header.index('rel')
            docid_idx= header.index('docid')
            for line in tqdm(reader): 
                query = line[query_idx]
                rel = int(line[rel_idx])
                docid = int(line[docid_idx])
                queries_to_judgements[query] = queries_to_judgements.get(query, []) + [(docid, rel)]
    return queries_to_judgements


def run_relevance_tests(queries_to_judgements: dict, outfile:str, ranker, cutoff:int = 10) -> dict[str, float]:
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename [str]: The filename containing the relevance data to be loaded

        ranker: A ranker configured with a particular scoring function to search through the document collection.
                This is probably either a Ranker or a L2RRanker object, but something that has a query() method

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    maps = []
    ncdgs = []
    for query, relevance_ratings in tqdm(list(queries_to_judgements.items())):
        search_results = ranker.query(query)
        map_relevance_scores = []
        ndcg_relevance_scores = []
        # add zero for ones 
        if len(search_results) == 0:
            maps.append(0)
            ncdgs.append(0)
            continue
        for result in search_results[:cutoff]:
            if len(result) == 0:
                continue
            result_docid = result[0]
            found = False
            for tup in tqdm(relevance_ratings):
                docid, rel  = tup
                if int(docid) == int(result_docid):
                    found = True
                    if int(rel) >= 4:
                        map_relevance_scores.append(1)
                        ndcg_relevance_scores.append(1)
                        break
                    else:
                        map_relevance_scores.append(0)
                        ndcg_relevance_scores.append(0)
                        break
            if not found:
                map_relevance_scores.append(0)
                ndcg_relevance_scores.append(0)

        map = map_score(map_relevance_scores, cutoff)
        ncdg = ndcg_score(map_relevance_scores, sorted(ndcg_relevance_scores, reverse = True), cutoff)

        maps.append(map)
        ncdgs.append(ncdg)

        with open(outfile, "a") as e:
            writer = csv.writer(e, delimiter=",")
            r1 = [query, 'map', map]
            r2 = [query, 'ndcg', ncdg]
            writer.writerow(r1)
            writer.writerow(r2)
    return {'map': np.mean(maps), 'ndcg': np.mean(ncdgs)}



# TODO (HW5): Implement NFaiRR metric for a list of queries to measure fairness for those queries
# NOTE: This has no relation to relevance scores and measures fairness of representation of classes
def run_fairness_test(attributes_file_path: str, protected_class: str, queries: list[str],
                      ranker, cut_off: int = 200) -> float:
    """
    Measures the fairness of the IR system using the NFaiRR metric.

    Args:
        attributes_file_path: The filename containing the documents about people and their demographic attributes
        protected_class: A specific protected class (e.g., Ethnicity, Gender)
        queries: A list containing queries
        ranker: A ranker configured with a particular scoring function to search through the document collection
        cut_off: The rank cut-off to use for calculating NFaiRR

    Returns:
        The average NFaiRR score across all queries
    """
    # TODO (HW5): Load person-attributes.csv

    doc_to_attributes = {}
    attribute_docs = []
    with open(attributes_file_path, 'r') as f:
        #Ethnicity,Gender,Religious_Affiliation,Political_Party,docid
        reader = csv.reader(f)
        attribute_idx = reader.index(protected_class)
        reader = next(reader)
        for row in reader:
            docid = row[-1]
            doc_to_attributes[docid] = row[:-1]
            attribute_docs.append(row[attribute_idx])

    scores = []
    for query in tqdm(queries):
        results = ranker.query(query)
        if len(results) == 0:
            #SOMETHING HERE 
            continue 
        end = min(cut_off, len(results))  
        for i, result in enumerate(results[:end]): 
            if len(result) == 0:
                continue 
            docid = result[0]
            result_attributes = doc_to_attributes[docid]
            if result_attributes is not None: 
                score = 1 / np.log(i + 1)
                scores.append(score)
            else:
                scores.append(0)

    return np.sum(scores)

        
                


    # TODO (HW5): Find the documents associated with the protected class

    # TODO (HW5): Loop through the queries and
    #       1. Create the list of omega values for the ranked list.
    #       2. Compute the NFaiRR score
    # NOTE: This fairness metric has some 'issues' (and the assignment spec asks you to think about it)

    
