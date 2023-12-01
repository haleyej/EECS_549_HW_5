from tqdm import tqdm
import pandas as pd
import lightgbm
from indexing import InvertedIndex
import multiprocessing
from collections import defaultdict, Counter
import numpy as np
from document_preprocessor import Tokenizer
from ranker import Ranker, TF_IDF, BM25, PivotedNormalization, CrossEncoderScorer
from vector_ranker import VectorRanker
import csv


# TODO: scorer has been replaced with ranker in initialization, check README for more details
class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker system.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object
            feature_extractor: The L2RFeatureExtractor object
        """
        # TODO: Save any new arguments that are needed as fields of this class

        # TODO: Initialize the LambdaMART model (but don't train it yet)
        self.doc_index = document_index 
        self.title_index = title_index 
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords 
        self.ranker = ranker
        self.feature_extractor = feature_extractor
        self.model = LambdaMART()
                   
    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores: A dictionary of queries mapped to a list of
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            X (list): A list of feature vectors for each query-document pair
            y (list): A list of relevance scores for each query-document pair
            qgroups (list): A list of the number of documents retrieved for each query
        """
        # NOTE: qgroups is not the same length as X or y
        #       This is for LightGBM to know how many relevance scores we have per query
        X = []
        y = []
        qgroups = []

        for query, ratings in list(query_to_document_relevance_scores.items()):
            query_parts = set(self.document_preprocessor.tokenize(query))
            title_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)
            doc_counts = self.accumulate_doc_term_counts(self.doc_index, query_parts)

            qgroups.append(len(ratings))
            for rating in tqdm(ratings): 
                docid, relevance = rating 
                doc_count = doc_counts[docid]
                title_count = title_counts[docid]
                features = self.feature_extractor.generate_features(docid, doc_count, title_count, query_parts, query)
                X.append(features)
                y.append(relevance)
                
        return X, y, qgroups

        # TODO: For each query and the documents that have been rated for relevance to that query,
        #       process these query-document pairs into features

            # TODO: Accumulate the token counts for each document's title and content here

            # TODO: For each of the documents, generate its features, then append
            #       the features and relevance score to the lists to be returned

            # Make sure to keep track of how many scores we have for this query

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        # TODO: Retrieve the set of documents that have each query word (i.e., the postings) and
        #       create a dictionary that keeps track of their counts for the query word
        term_freqs = defaultdict(lambda : {})

        query_parts = set(query_parts)
        for part in query_parts: 
            postings = index.get_postings(part)
            if postings == None or len(postings) == 0:
                break
            for posting in postings: 
                docid = posting[0]
                count = posting[1]
                term_freqs[docid].update({part: count})
        return term_freqs

    def train(self, training_data_filename: str) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
        """
        # TODO: Convert the relevance data into the right format for training data preparation
        
        # TODO: Prepare the training data by featurizing the query-doc pairs and
        #       getting the necessary datastructures
        
        # TODO: Train the model
        query_doc_pairs = {}
        with open(training_data_filename) as f:
            reader = csv.reader(f)
            header = next(reader)
            query_idx = header.index("query")
            docid_idx = header.index("docid")
            rel_idx = header.index("rel")
            for doc in tqdm(reader):
                query = doc[query_idx]
                docid = int(doc[docid_idx])
                rel = int(doc[rel_idx])
                curr = query_doc_pairs.get(query, [])
                curr.append((docid, rel))
                query_doc_pairs[query] = curr 
        X, y, qgroups = self.prepare_training_data(query_doc_pairs) 

        tqdm(self.model.fit(X, y, qgroups))


    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        # TODO: Return a prediction made using the LambdaMART model
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    # TODO (HW5): Implement MMR diversification for a given list of documents and their cosine similarity scores
    @staticmethod
    def maximize_mmr(thresholded_search_results: list[tuple[int, float]], similarity_matrix: np.ndarray,
                     list_docs: list[int], mmr_lambda: int) -> list[tuple[int, float]]:
        """
        Takes the thresholded list of results and runs the maximum marginal relevance diversification algorithm
        on the list.
        It should return a list of the same length with the same overall documents but with different document ranks.
        
        Args:
            thresholded_search_results: The thresholded search results
            similarity_matrix: Precomputed similarity scores for all the thresholded search results
            list_docs: The list of documents following the indexes of the similarity matrix
                       If document 421 is at the 5th index (row, column) of the similarity matrix,
                       it should be on the 5th index of list_docs.
            mmr_lambda: The hyperparameter lambda used to measure the MMR scores of each document

        Returns:
            A list containing tuples of the ranked documents and their scores
        """
        # NOTE: This algorithm implementation requires some amount of planning as you need to maximize
        #       the MMR at every step.
        #       1. Create an empty list S
        #       2. Find the element with the maximum MMR in thresholded_search_results, R (but not in S)
        #       3. Move that element from R and append it to S
        #       4. Repeat 2 & 3 until there are no more remaining elements in R to be processed

        S = []
        R = thresholded_search_results.copy()

        while len(list_docs) > 0:
            top_mmr_score = 0
            top_mmr_docid = None
            top_mmr_tup = tuple()
            for i, tup in enumerate(R):
                docid, rel_score = tup
                max_sim = 0
                for j in S: 
                    for j, result in enumerate(S): 
                        sim = similarity_matrix[i, j]
                        if sim > max_sim:
                            max_sim = sim
                mmr_score = (mmr_lambda * rel_score) - ((1 - mmr_lambda) * max_sim)
                print('CURRENT MMR SCORE', mmr_score)
                if mmr_score > top_mmr_score:
                    top_mmr_score = mmr_score
                    top_mmr_docid = docid
                    top_mmr_tup = tup

            print('TOP MMR DOC', top_mmr_docid)
            S.append((top_mmr_docid, top_mmr_score))
            if top_mmr_docid ==  None: 
                return S
            list_docs.remove(top_mmr_docid)
            R.remove(top_mmr_tup)

        return S
    
    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2, user_id=None, mmr_lambda:int=1, mmr_threshold:int=100) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number of top-ranked documents
                to be used in the query
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query
            user_id: the integer id of the user who is issuing the query or None if the user is unknown
            mmr_lambda: Hyperparameter for MMR diversification scoring
            mmr_threshold: Documents to rerank using MMR diversification

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        # TODO: Retrieve potentially-relevant documents
        
        # TODO: Fetch a list of possible documents from the index and create a mapping from
        #       a document ID to a dictionary of the counts of the query terms in that document.
        #       You will pass the dictionary to the RelevanceScorer as input
        #
        # NOTE: we collect these here (rather than calling a Ranker instance) because we'll
        #       pass these doc-term-counts to functions later, so we need the accumulated representations

        # TODO: Accumulate the documents word frequencies for the title and the main body

        # TODO: Score and sort the documents by the provided scorer for just the document's main text (not the title).
        #       This ordering determines which documents we will try to *re-rank* using our L2R model

        # TODO: Filter to just the top 100 documents for the L2R part for re-ranking

        # TODO: Construct the feature vectors for each query-document pair in the top 100

        # TODO: Use your L2R model to rank these top 100 documents
        
        # TODO: Sort posting_lists based on scores
        
        # TODO: Make sure to add back the other non-top-100 documents that weren't re-ranked

        # TODO (HW5): Run MMR diversification for appropriate values of lambda

        # TODO (HW5): Get the threholded part of the search results, aka top t results and
        #      keep the rest separate

        # TODO (HW5): Get the document similarity matrix for the thresholded documents using vector_ranker
        #      Preserve the input list of documents to be used in the MMR function
        
        # TODO (HW5): Run the maximize_mmr function with appropriate arguments
        
        # TODO (HW5): Add the remaining search results back to the MMR diversification results
        
        # TODO: Return the ranked documents
        query_parts = self.document_preprocessor.tokenize(query)
        query_parts = [part for part in query_parts if part not in self.stopwords or part != None]
        doc_word_counts = self.accumulate_doc_term_counts(self.doc_index, query_parts)
        candidates = set(list(doc_word_counts.keys()))
        title_word_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)
        title_word_counts =  {k:v for k,v in title_word_counts.items() if k in candidates}

        scores = self.ranker.query(query, pseudofeedback_num_docs, pseudofeedback_alpha, pseudofeedback_beta, user_id)
        candidates_and_scores = sorted(scores, key = lambda s: s[1], reverse = True)

        if len(candidates_and_scores) > 100:
            rerank = candidates_and_scores[:100]
        else:
            rerank = candidates_and_scores.copy()

        docs_to_predictions = []
        for docid, _ in tqdm(rerank):
            doc_word_count = doc_word_counts.get(docid, {})
            title_word_count = title_word_counts.get(docid, {})
            feature_vec = self.feature_extractor.generate_features(docid, doc_word_count, title_word_count, query_parts, query)
            prediction = self.model.predict(np.array(feature_vec).reshape(1, -1))[0]
            docs_to_predictions.append((docid, prediction))


        sorted_predictions = sorted(docs_to_predictions, key = lambda s: s[1], reverse = True)
        
        combined_results = sorted_predictions + [(d, s) for d, s in candidates_and_scores[100:]]

        if mmr_threshold > 0:
            mmr_docs = [d for d, s in combined_results[:mmr_threshold]]
            similarity_matrix = self.ranker.document_similarity(mmr_docs)
            mmr_results = self.maximize_mmr(combined_results[:mmr_threshold], similarity_matrix, mmr_docs, mmr_lambda)

            results = mmr_results + combined_results[mmr_threshold:]
            return results

        
        return combined_results


class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 doc_category_info: dict[int, list[str]],
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 recognized_categories: set[str], docid_to_network_features: dict[int, dict[str, float]],
                 ce_scorer: CrossEncoderScorer) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
            ce_scorer: The CrossEncoderScorer object
        """
        self.doc_index = document_index 
        self.title_index = title_index 
        self.doc_category_info = doc_category_info 
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.recognized_categories = recognized_categories
        self.docid_to_network_features = docid_to_network_features

        #scorers 
        self.BM25 = BM25(self.doc_index)
        self.pivoted_norm = PivotedNormalization(self.doc_index)
        self.doc_tf_idf = TF_IDF(self.doc_index)
        self.title_tf_idf = TF_IDF(self.title_index)
        self.ce_scorer = ce_scorer

    # TODO: Article Length
    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.doc_index.get_doc_metadata(docid)['total_token_count']

    # TODO: Title Length
    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.title_index.get_doc_metadata(docid)['total_token_count']

    # TODO: TF
    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        scores = []
        mutual_terms = set(query_parts) & set(word_counts.keys()) 
        for term in mutual_terms:
            tf = word_counts[term]
            scores.append(np.log(tf + 1))
        score = np.sum(scores)
        return score

    # TODO: TF-IDF
    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        query_word_counts = Counter(query_parts)
        if index == self.title_index: 
            return self.title_tf_idf.score(docid, word_counts, query_word_counts)
        elif index == self.doc_index:
            return self.doc_tf_idf.score(docid, word_counts, query_word_counts)

    # TODO: BM25
    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        query_word_counts = Counter(query_parts)
        return self.BM25.score(docid, doc_word_counts, query_word_counts)

    # TODO: Pivoted Normalization
    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        query_word_counts = Counter(query_parts)
        return self.pivoted_norm.score(docid, doc_word_counts, query_word_counts)

    # TODO: Document Categories
    def get_document_categories(self, docid: int) -> list:
        """
        Generates a list of binary features indicating which of the recognized categories that the document has.
        Category features should be deterministically ordered so list[0] should always correspond to the same
        category. For example, if a document has one of the three categories, and that category is mapped to
        index 1, then the binary feature vector would look like [0, 1, 0].

        Args:
            docid: The id of the document

        Returns:
            A list containing binary list of which recognized categories that the given document has
        """
        category_vec = []
        doc_categories = self.doc_category_info.get(docid, [])
        for category in self.recognized_categories:
            if category in doc_categories:
                category_vec.append(1)
            else:
                category_vec.append(0)
        return category_vec

    def get_pagerank_score(self, docid: int) -> float:
        """
        Gets the PageRank score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The PageRank score
        """
        network_features = self.docid_to_network_features.get(docid, {})
        if len(network_features) == 0:
            return 0
        return network_features['pagerank']

    # TODO: HITS Hub
    def get_hits_hub_score(self, docid: int) -> float:
        """
        Gets the HITS hub score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS hub score
        """
        network_features = self.docid_to_network_features.get(docid, {})
        if len(network_features) == 0:
            return 0
        return network_features['hub_score']

    # TODO: HITS Authority
    def get_hits_authority_score(self, docid: int) -> float:
        """
        Gets the HITS authority score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS authority score
        """
        network_features = self.docid_to_network_features.get(docid, {})
        if len(network_features) == 0:
            return 0
        return network_features['authority_score']

    # TODO (HW3): Cross-Encoder Score
    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """        
        return self.ce_scorer.score(docid, query)

    # TODO: Add at least one new feature to be used with your L2R model

    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str],
                          query: str) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """
        # NOTE: We can use this to get a stable ordering of features based on consistent insertion
        #       but it's probably faster to use a list to start

        doc_metadata = self.doc_index.get_doc_metadata(docid)
        if len(doc_metadata) == 0:
            doc_len = 0
        else:
            doc_len = doc_metadata['total_token_count']
        
        title_metadata = self.title_index.get_doc_metadata(docid)
        if len(title_metadata) == 0:
            title_len = 0
        else:
            title_len = title_metadata['total_token_count']

        query_len = len(query_parts)

        doc_tf = self.get_tf(self.doc_index, docid, doc_word_counts, query_parts)

        doc_tf_idf = self.get_tf_idf(self.doc_index, docid, doc_word_counts, query_parts)
        
        title_tf = self.get_tf(self.title_index, docid, title_word_counts, query_parts)

        title_tf_idf = self.get_tf_idf(self.title_index, docid, title_word_counts, query_parts)
        
        doc_bm25 = self.get_BM25_score(docid, doc_word_counts, query_parts)

        doc_pivoted_norm = self.get_pivoted_normalization_score(docid, doc_word_counts, query_parts)

        doc_page_rank = self.get_pagerank_score(docid)

        doc_hub_score = self.get_hits_hub_score(docid)

        doc_authority_score = self.get_hits_authority_score(docid)

        doc_cross_encoder_score = self.get_cross_encoder_score(docid, query)

        if len(doc_metadata) == 0:
            uniquenes_ratio = 0
        else:
            uniquenes_ratio = doc_metadata['num_unique_tokens'] / doc_metadata['total_token_count']
 
        doc_categories = self.get_document_categories(docid)  

        feature_vector = [doc_len, 
                          title_len, 
                          query_len,
                          doc_tf, 
                          doc_tf_idf, 
                          title_tf, 
                          title_tf_idf, 
                          doc_bm25, 
                          doc_pivoted_norm, 
                          doc_page_rank, 
                          doc_hub_score, 
                          doc_authority_score,
                          doc_cross_encoder_score,
                          uniquenes_ratio] 

        feature_vector.extend(doc_categories)
        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 20,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.005,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to
            # the number of CPUs on your machine for faster training
            "n_jobs": multiprocessing.cpu_count()-1,
            "verbosity": 1,
        }

        if params:
            default_params.update(params)

        self.model = lightgbm.LGBMRanker(boosting_type = default_params['boosting_type'], 
                                         num_leaves = default_params['num_leaves'],
                                         max_depth = default_params['max_depth'],
                                         learning_rate = default_params['learning_rate'],
                                         n_estimators = default_params['n_estimators'],
                                         objective = default_params['objective'],
                                         n_jobs = default_params['n_jobs'],
                                         importance_type = default_params['importance_type'],
                                         metric = default_params['metric']
                                         )
  
    def fit(self, X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples
            y_train (array-like): Target values
            qgroups_train (array-like): Query group sizes for training data

        Returns:
            self: Returns the instance itself
        """
        # TODO: Fit the LGBMRanker's parameters using the provided features and labels
        self.model.fit(X_train, y_train, group = qgroups_train)
        return self

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """

        return self.model.predict(featurized_docs)

