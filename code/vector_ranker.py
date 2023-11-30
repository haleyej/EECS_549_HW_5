from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
from ranker import Ranker
import numpy as np
from itertools import combinations


class VectorRanker(Ranker):
    def __init__(self, bi_encoder_model_name: str, encoded_docs: ndarray,
                 row_to_docid: list[int]) -> None:
        """
        Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoded
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding

        Using zip(encoded_docs, row_to_docid) should give you the mapping between the docid and the embedding.
        """
        # TODO: Instantiate the bi-encoder model here

        # NOTE: we're going to use the cpu for everything here so if you decide to use a GPU, do not 
        # submit that code to the autograder
        self.bi_encoder_model = SentenceTransformer(bi_encoder_model_name, device = 'cpu')
        self.encoded_docs = encoded_docs
        self.row_to_docid = row_to_docid
        #docids = [docid for docid, _ in ]
        self.mapping = dict(zip(row_to_docid, encoded_docs))
        

    def query(self, query: str, pseudofeedback_num_docs=0,
              pseduofeedback_alpha=0.8, pseduofeedback_beta=0.2, user_id = None) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.
        Performs query expansion using pseudo-relevance feedback if needed.

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number of top-ranked documents
                to be used in the query
            pseduofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseduofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first
        """
        encoded_query = self.bi_encoder_model.encode(query)
        scores = util.dot_score(encoded_query, self.encoded_docs)[0].cpu().tolist()
        scores_and_ids = []
        [scores_and_ids.append((tup, scores[i])) for i, tup in enumerate(self.row_to_docid)]
        scores_and_ids = sorted(scores_and_ids, key = lambda s: s[1], reverse = True)


        if pseudofeedback_num_docs > 0:
            rel_docs = scores_and_ids[:pseudofeedback_num_docs]
            rel_doc_vecs = np.array([self.mapping[docid] for docid, _ in rel_docs])
            avg_doc_vec = np.mean(rel_doc_vecs, axis = 0)
            modified_query = (pseduofeedback_alpha * encoded_query) + (pseduofeedback_beta * avg_doc_vec)

            scores = util.dot_score(modified_query, self.encoded_docs)[0].cpu().tolist()
            scores_and_ids = []
            [scores_and_ids.append((tup, scores[i])) for i, tup in enumerate(self.row_to_docid)]
            scores_and_ids = sorted(scores_and_ids, key = lambda s: s[1], reverse = True)

        return scores_and_ids


    # TODO (HW5): Find the dot product (unnormalized cosine similarity) for the list of documents (pairwise)
    # NOTE: You should return a matrix where element [i][j] would represent similarity between
    #   list_docs[i] and list_docs[j]
    def document_similarity(self, list_docs: list[int]) -> np.ndarray:
        """
        Calculates the pairwise similarities for a given list of documents

        Args:
            list_docs: A list of document IDs

        Returns:
            A matrix where element [i][j] is a similarity score between list_docs[i] and list_docs[j]
        """
        dot_scores = []
        for doc_1 in list_docs:
            for doc_2 in list_docs:
                doc_1_embed = self.mapping[doc_1]
                doc_2_embed = self.mapping[doc_2]
                dot_score = util.dot_score(doc_1_embed, doc_2_embed)[0].cpu().tolist()
                dot_scores.append(dot_score[0])
        return np.array(dot_scores).reshape(len(list_docs), len(list_docs))

        

