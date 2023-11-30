## General strategy for Assignment 5
Homework 5 will have you extending the code from Homework 4 to work with bias and diversity
1. Measuring Fairness in Ranking: implement NFaiRR
2. Increasing Diversity in Ranking: implement Maximal Marginal Relevance (MMR) 

## Changes from Assignment 4

### `relevance.py`

- New function `nfairr_score` which computes the normalized Fairness of Retrieval Results (NFAIRR) score for a list of omega values for the list of ranked documents
  - If all documents are from the protected class, then the NFAIRR score is 0
  - Arguments:
    - `actual_omega_values`: The omega value for a ranked list of documents. The most relevant document is the first item in the list.
    - `cut_off`: The rank cut-off to use for calculating NFAIRR. Omega values in the list after this cut-off position are not used. The default is 200.
  - Returns the NFAIRR score
- New function `run_fairness_test`
  - Implement NFaiRR metric for a list of queries to measure fairness for those queries
  - NOTE: This has no relation to relevance scores and measures fairness of representation of classes 
  - Arguments:
    - `attributes_file_path`: The path where `person-attributes.csv` is saved
    - `protected_class`: The protected class (e.g., Race)
    - `queries`: The list of queries
    - `ranker`: a Ranker
    - `cutoff`: The rank cut-off to use for calculating NFAIRR

---

### `l2r.py`
- In the `L2RRanker` class there is a new function `maximize_mmr` that takes the thresholded list of results and runs the maximum marginal relevance diversification algorithm on the list
  - Arguments:
    - `thresholded_search_results`: The thresholded search results
    - `similarity_matrix`: Precomputed similarity scores for all the thresholded search results
    - `list_docs`: The list of documents following the indexes of the similarity matrix.
      - If document 421 is at the 5th index (row, column) of the similarity matrix,
      - it should be on the 5th index of list_docs
    - `mmr_lambda`: The hyperparameter lambda used to measure the MRR scores of each document.
  - Returns a list of the same length with the same overall documents but with different document ranks
- In `query`, run MRR diversification for appropriate values of lambda by calling `maximize_mmr` to rerank at the very end
---

### `vector_ranker.py`

- New function `document_similarity` to compute the `similarity_matrix` in `L2RRanker.maximize_mmr`
  - Can use the dot product here, since the vectors are normalized for the default models we use
  - Return a matrix (np.ndarray) where element [i][j] represents the similarity between list_docs[i] and list_docs[j]

---

## How to use the public test cases

- To run individual test cases, in your terminal, run:
  * `python [filename] [class].[function]`
  * ex: `python test_relevance_scorers.py TestRankingMetrics.test_bm25_single_word_query`
 
- To run one class's tests from file, in your terminal, run:
  * `python [filename] [class] -vvv`
  * ex: `python test_indexing.py TestBasicInvertedIndex -vvv`

- To run all the tests from file, in your terminal, run:
  * `python [filename] -vvv`
  * ex: `python test_indexing.py -vvv`


- To add your own test cases, the basic structure is as follows:
  
```
import unittest

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
  
```
More information can be found [here](https://docs.python.org/3/library/unittest.html).
