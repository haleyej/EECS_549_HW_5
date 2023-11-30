import math
import csv
from tqdm import tqdm
import pandas as pd


# TODO (HW5): Implement NFaiRR
def nfairr_score(actual_omega_values: list[int], cut_off=200) -> float:
    """
    Computes the normalized fairness-aware rank retrieval (NFaiRR) score for a list of omega values
    for the list of ranked documents.
    If all documents are from the protected class, then the NFaiRR score is 0.

    Args:
        actual_omega_values: The omega value for a ranked list of documents
            The most relevant document is the first item in the list.
        cut_off: The rank cut-off to use for calculating NFaiRR
            Omega values in the list after this cut-off position are not used. The default is 200.

    Returns:
        The NFaiRR score
    """
    # TODO (HW5): Compute the FaiRR and IFaiRR scores using the given list of omega values
    # TODO (HW5): Implement NFaiRR
    pass


def map_score(search_result_relevances: list[int], cut_off=10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    # TODO: Implement MAP
    pass


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_off=10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: Implement NDCG
    pass


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance dataset

    # TODO: Run each of the dataset's queries through your ranking function

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out

    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance 
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.

    # NOTE: NDCG can use any scoring range, so no conversion is needed.
  
    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    return {'map': 0, 'ndcg': 0}


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

    # TODO (HW5): Find the documents associated with the protected class

    score = []

    # TODO (HW5): Loop through the queries and
    #       1. Create the list of omega values for the ranked list.
    #       2. Compute the NFaiRR score
    # NOTE: This fairness metric has some 'issues' (and the assignment spec asks you to think about it)

    pass
    
