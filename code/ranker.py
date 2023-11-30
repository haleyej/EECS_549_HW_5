"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization,
and your own ranker.
"""
import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import CrossEncoder
from indexing import InvertedIndex
import math

from indexing import Indexer, IndexType
from document_preprocessor import RegexTokenizer
import json


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str], 
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int,str]) -> None:
        """
        Initializes the state of the Ranker object.

        NOTE: Previous homeworks had you passing the class of the scorer to this function.
            This has been changed as it created a lot of confusion.
            You should now pass an instantiated RelevanceScorer to this function.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict


    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2, user_id=None) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number
                 of top-ranked documents to be used in the query,
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query
            user_id: The integer id of the user who is issuing the query or None if the user is unknown

        Returns:
            A list containing tuples of the documents (ids) and their relevance scores

        NOTE: We are standardizing the query output of Ranker to match with L2RRanker.query and VectorRanker.query
        The query function should return a sorted list of tuples where each tuple has the first element as the document ID
        and the second element as the score of the document after the ranking process.
        """
        scores = []
        query_parts = self.tokenize(query)
        query_word_counts = Counter(query_parts) 
        candidate_docs = {}

        for part in set(query_parts):
            doc_count = {}
            if part.lower() in self.stopwords:
                continue
            postings = self.index.get_postings(part)
            if postings == None:
                continue
            for docid, count in postings: 
                doc_count[part] = count
                candidate_docs[docid] = candidate_docs.get(docid, {}) | doc_count

        for docid in candidate_docs:
            counts = candidate_docs[docid]
            score = self.scorer.score(docid = docid, doc_word_counts = counts, query_word_counts = query_word_counts)
            scores.append((docid, score))

        scores = sorted(scores, key = lambda s: s[1], reverse = True)


        if pseudofeedback_num_docs > 0:
            pseudo_doc_counts = Counter()
            for docid, _ in scores[:pseudofeedback_num_docs]:
                counts = candidate_docs[docid]
                tokens = self.tokenize(self.raw_text_dict[docid])
                tokens = [t.lower() for t in tokens if t.lower() not in self.stopwords]
                pseudo_doc_counts.update(tokens)

   
            words = set(query_word_counts.keys()).union(set(pseudo_doc_counts.keys()))

            weighted_query = {}
            for word in words:
               weighted_query[word] = (pseudofeedback_alpha * query_word_counts[word]) + (pseudofeedback_beta / pseudofeedback_num_docs) * pseudo_doc_counts[word]

            modified_candidate_docs = {}
            for part in set(weighted_query.keys()):
                postings = self.index.get_postings(part)
                if postings == None or len(postings) == 0:
                    continue
                doc_count = {}
                for docid, count in postings: 
                    doc_count[part] = count
                    modified_candidate_docs[docid] = modified_candidate_docs.get(docid, {}) | doc_count
                        

            modified_scores = []
            for docid, counts in list(modified_candidate_docs.items()):
                modified_score = self.scorer.score(docid = docid, doc_word_counts = counts, query_word_counts = weighted_query)
                modified_scores.append((docid, modified_score))

            modified_scores = sorted(modified_scores, key = lambda s: s[1], reverse = True)
            return modified_scores
        else:
            return scores 


class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """

    def __init__(self, index: InvertedIndex, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        TODO (HW4): Note that the `query_word_counts` is now a dictionary of words and their counts.
            This is changed from the previous homeworks.
        """
        raise NotImplementedError


# TODO: Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query

        # 2. Return the score
        scores = []
        for part in list(query_word_counts.keys()):
            if part in list(doc_word_counts.keys()):
                scores.append(doc_word_counts[part] * query_word_counts[part])
    
        if len(scores) == 0:
            return None 
        score = np.sum(scores)
        if score == 0:
            return None
        return score 


# TODO: Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters
        self.num_tokens = index.get_statistics()['total_token_count']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]):
        doc_word_counts 

        d = self.index.get_doc_metadata(docid)['total_token_count']
        q = len(list(query_word_counts.keys()))
        mu = self.parameters['mu']

        scores = []
        mutual_terms = set(doc_word_counts.keys()) & set(query_word_counts.keys())
        for k in mutual_terms:
            query_count = query_word_counts[k]
            doc_count = doc_word_counts[k]
            prob = self.index.get_term_metadata(k)['count'] / self.num_tokens

            score = query_count * np.log(1 + (doc_count / (mu * prob)))

            scores.append(score)

        length_norm_ish_term = np.log(mu / (d + mu))
        score = np.sum(scores)
        if score == 0:
            return None
        return score + (q * length_norm_ish_term)
    
# TODO: Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.N = index.get_statistics()['number_of_documents']
        self.avg_d = index.get_statistics()['mean_document_length']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        doc_freq_counts = Counter()

        for part in list(query_word_counts.keys()):
            term_metadata = self.index.get_term_metadata(part)
            if len(list(term_metadata.keys())) == 0:
                continue 
            doc_freq_counts[part] = term_metadata['n_docs']

        doc_metadata = self.index.get_doc_metadata(docid)
        if len(doc_metadata) == 0:
            d = 0
        else:
            d = self.index.get_doc_metadata(docid)['total_token_count']
        b = self.b
        k1 = self.k1
        k3 = self.k3

        scores = []
        mutual_terms = set(doc_word_counts.keys()) & set(query_word_counts.keys())
        for k in mutual_terms:
            query_count = query_word_counts[k]
            doc_count = doc_word_counts[k]
            doc_freq = doc_freq_counts[k]
            idf_ish_term = np.log((self.N - doc_freq + 0.5) / (doc_freq + 0.5))
            tf_ish_term = ((k1 + 1) * doc_count) / ((k1 * (1 - b + (b *(d/self.avg_d)))) + doc_count)
            qtf_ish_term = ((k3 + 1) * query_count) / (k3 + query_count)
            score = idf_ish_term * tf_ish_term * qtf_ish_term 
            scores.append(score)

        return np.sum(scores)


class PersonalizedBM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, relevant_doc_index: InvertedIndex,
                 parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        """
        Initializes Personalized BM25 scorer.

        Args:
            index: The inverted index used to use for computing most of BM25
            relevant_doc_index: The inverted index of only documents a user has rated as relevant,
                which is used when calculating the personalized part of BM25
            parameters: The dictionary containing the parameter values for BM25

        Returns:
            The Personalized BM25 score
        """
        self.index = index
        self.relevant_doc_index = relevant_doc_index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.N = index.get_statistics()['number_of_documents']
        self.avg_d = index.get_statistics()['mean_document_length']

        self.R = relevant_doc_index.get_statistics()['number_of_documents']


    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        b = self.b
        k1 = self.k1 
        k3 = self.k3 
        N = self.N
        avg_d = self.avg_d 
        R = self.R

        d = self.index.get_doc_metadata(docid).get('total_token_count', 0)
        mutual_terms = set(doc_word_counts.keys()) & set(query_word_counts.keys())
        scores = []

        for part in mutual_terms:
            doc_freq = self.index.get_term_metadata(part).get('n_docs', 0)
            rel_doc_freq = self.relevant_doc_index.get_term_metadata(part).get('n_docs', 0)

            query_count = query_word_counts[part]
            doc_count = doc_word_counts[part]

            idf_ish_term_num = (rel_doc_freq + 0.5) * (N - doc_freq - R + rel_doc_freq + 0.5)
            idf_ish_term_denom = (doc_freq - rel_doc_freq + 0.5) * (R - rel_doc_freq + 0.5)
            idf_ish_term = np.log(idf_ish_term_num / idf_ish_term_denom)


            tf_ish_term = ((k1 + 1) * doc_count) / ((k1 * ((1 - b) + b * (d / avg_d))) + doc_count)
            qtf_ish_term = ((k3 + 1) * query_count) / (k3 + query_count)

            score = idf_ish_term * tf_ish_term * qtf_ish_term
            scores.append(score)

        return np.sum(scores)
            

class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']
        self.N = index.get_statistics()['number_of_documents']
        self.avg_d = index.get_statistics()['mean_document_length']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        doc_freq_counts = Counter()

        for part in list(query_word_counts.keys()):
            term_metadata = self.index.get_term_metadata(part)
            if len(list(term_metadata.keys())) == 0:
                continue 
            doc_freq_counts[part] = term_metadata['n_docs']

        doc_metadata = self.index.get_doc_metadata(docid)
        if len(doc_metadata) == 0:
            d = 0
        else:
            d = self.index.get_doc_metadata(docid)['total_token_count']
        b = self.b

        scores = []
        mutual_terms = set(doc_word_counts.keys()) & set(query_word_counts.keys())

        for k in mutual_terms:
            query_count = query_word_counts[k]
            doc_count = doc_word_counts[k]
            doc_freq = doc_freq_counts[k]

            tf_ish_term = 1 + np.log(1 + np.log(doc_count))
            length_norm_term = 1 - b + (b * (d / self.avg_d))
            idf_ish_term = np.log((self.N + 1) / doc_freq)

            score = query_count * (tf_ish_term / length_norm_term) * idf_ish_term

            scores.append(score)

        score = np.sum(scores)
        return score


class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters
        self.N = index.get_statistics()['number_of_documents']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts:dict = None):
        query_parts = list(query_word_counts.keys())
        doc_freq_counts = Counter()

        for part in set(query_parts):
            term_metadata = self.index.get_term_metadata(part)
            if len(list(term_metadata.keys())) == 0:
                continue 
            doc_freq_counts[part] = term_metadata['n_docs']

        mutual_terms = set(doc_word_counts.keys()) & set(query_parts)

        doc_tf_idf = np.zeros(len(mutual_terms))
        for i, k in enumerate(mutual_terms):
            doc_count = doc_word_counts[k]
            doc_freq = doc_freq_counts[k]

            tf_term = np.log(doc_count + 1)
            idf_term = 1 + np.log(self.N /doc_freq)
            doc_tf_idf[i] = (tf_term * idf_term)
            
        if len(doc_tf_idf) == 0:
            return 0
        score = np.sum(doc_tf_idf)
        return score


class CrossEncoderScorer:
    def __init__(self, raw_text_dict: dict[int, str],
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model

        NOTE 1: The CrossEncoderScorer class uses a pre-trained cross-encoder model
            from the Sentence Transformers package to score a given query-document pair.

        NOTE 2: This is not a RelevanceScorer object because the method signature for score() does not match,
            but it has the same intent, in practice.
        """
        self.raw_text_dict = raw_text_dict
        self.encoder = CrossEncoder(cross_encoder_model_name, max_length = 512)

    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.
        
        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        text = self.raw_text_dict.get(docid, '')
        if len(text) == 0 or len(query) == 0:
            return 0
        score = self.encoder.predict((query, text))
        return score

class YourRanker(RelevanceScorer):
    def __init__(self, index, parameters: dict = {}):
        self.index = index 
        self.parameters = parameters 
        self.N = index.get_statistics()['number_of_documents']
        self.avg_d = index.get_statistics()['mean_document_length']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> dict[str, int]:
        '''
        Uses index to score a query against a particular document 

        Returns a dictionary with the keys docid and score 

        ARGS:
            docid: 
                document id of the doc we want to score
            query_parts:
                tokenized list of words in the query, with
                stopwords removed and replaced with None

        EXAMPLE INPUT:
            docid = 1
            query_parts = ['University', None, 'Michigan']

        EXAMPLE OUTPUT:
            {'docid': 1, 'score': 0.5}
        '''
        doc_freq_counts = Counter()
        query_word_counts = Counter(query_parts)

        for part in set(query_parts):
            term_metadata = self.index.get_term_metadata(part)
            if len(list(term_metadata.keys())) == 0:
                continue 
            doc_freq_counts[part] = term_metadata['n_docs']

        d = self.index.get_doc_metadata(docid)['total_token_count']
        scores = []
        for k in doc_word_counts.keys():
            doc_word_count = doc_word_counts[k]
            doc_freq_count = doc_freq_counts[k]
            query_word_count = query_word_counts[k]

            tf_ish_term = np.log(np.log(doc_word_count + 1 / (d / self.avg_d)))
            idf_ish_term = np.log((self.N + 1) / doc_freq_count)
            qtf_ish_term = np.log(np.log(query_word_count + 1) + 1)

            score = query_word_count * tf_ish_term * idf_ish_term * qtf_ish_term
            scores.append(score)

        if len(scores) == 0:
            return None
        score = np.sum(scores)
        if score == 0:
            return None
        return score


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        # Print randomly ranked results
        return 10
    
