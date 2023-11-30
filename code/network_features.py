import numpy as np
from sknetwork.ranking import PageRank, HITS
import pandas as pd
from pandas import DataFrame
from tqdm.auto import tqdm
from sknetwork.data import from_edge_list
import gzip


class NetworkFeatures:
    """
    A class to help generate network features such as PageRank scores, HITS hub score and HITS authority scores.
    This class uses the scikit-network library https://scikit-network.readthedocs.io to calculate node ranking values.

    OPTIONAL reads
        1. PageRank: https://towardsdatascience.com/pagerank-algorithm-fully-explained-dc794184b4af
        2. HITS: https://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture4/lecture4.html
    """
    def load_network(self, network_filename: str, total_edges: int):
        """
        Loads the network from the specified file and returns the network. A network file 
        can be listed using a .csv or a .csv.gz file.

        Args:
            network_filename: The name of a .csv or .csv.gz file containing an edge list
            total_edges: The total number of edges in an edge list

        Returns:
            The loaded network from sknetwork
        """
        edges = []
        i = 0
        if network_filename.endswith(".gz"):
            with gzip.open(network_filename) as f:
                data = f.readline()
                while data:
                    i += 1
                    if i == 1:
                        data = f.readline()
                        continue
                    data = data.strip()
                    data = data.decode()
                    from_node, to_node = data.split(",")
                    edges.append((from_node, to_node))
                    data = f.readline()
        elif network_filename.endswith(".csv"):
            with open(network_filename) as f: 
                data = f.readline()
                while data:
                    i += 1
                    if i == 1:
                        data = f.readline()
                        continue
                    data = data.strip()
                    from_node, to_node = data.split(",")
                    if from_node == 'from' and to_node == 'to':
                        continue
                    edges.append((from_node, to_node))
                    data = f.readline()
        adjacency = from_edge_list(edges, directed = True)
        return adjacency
        

    def calculate_page_rank(self, graph, damping_factor=0.85, iterations=100, weights=None) -> list[float]:
        """
        Calculates the PageRank scores for the provided network and
        returns the PageRank values for all nodes.

        Args:
            graph: A graph from sknetwork
            damping_factor: The complement of the teleport probability for the random walker
                For example, a damping factor of .8 has a .2 probability of jumping after each step.
            iterations: The maximum number of iterations to run when computing PageRank
            weights: if Personalized PageRank is used, a data structure containing the restart distribution
                     as a vector (over the length of nodes) or a dict {node: weight}

        Returns:
            The PageRank scores for all nodes in the network (array-like)
        
        TODO (hw4): Note that `weights` is added as a parameter to this function for Personalized PageRank.
        """
        pagerank = PageRank(damping_factor = damping_factor, n_iter = iterations)
        if weights != None: 
            scores = pagerank.fit_predict(graph.adjacency, weights)
        else:
            scores = pagerank.fit_predict(graph.adjacency)
        
        return scores

    def calculate_hits(self, graph) -> tuple[list[float], list[float]]:
        """
        Calculates the hub scores and authority scores using the HITS algorithm
        for the provided network and returns the two lists of scores as a tuple.

        Args:
            graph: A graph from sknetwork

        Returns:
            The hub scores and authority scores (in that order) for all nodes in the network
        """
        # TODO: Use scikit-network to run HITS and return HITS hub scores and authority scores
        # NOTE: When returning the HITS scores, the returned tuple should have the hub scores in index 0 and
        #       authority score in index 1
        hits = HITS()
        hits.fit_predict(graph.adjacency)
        return (hits.scores_col_, hits.scores_row_)

    def get_all_network_statistics(self, graph) -> DataFrame:
        """
        Calculates the PageRank and the hub scores and authority scores using the HITS algorithm
        for the provided network and returns a pandas DataFrame with columns: 
        'docid', 'pagerank', 'authority_score', and 'hub_score' containing the relevant values
        for all nodes in the network.

        Args:
            graph: A graph from sknetwork

        Returns:
            A pandas DataFrame with columns 'docid', 'pagerank', 'authority_score', and 'hub_score'
            containing the relevant values for all nodes in the network
        """
        pagerank_scores = self.calculate_page_rank(graph)
        hub_scores, authority_scores = self.calculate_hits(graph)

        df = pd.DataFrame({
                            'docid': graph.names, 
                            'pagerank': pagerank_scores, 
                            'authority_score': authority_scores, 
                            'hub_score': hub_scores
                         })
        return df 


def main():
    from document_preprocessor import RegexTokenizer
    nf = NetworkFeatures()
    g = nf.load_network('../edgelist.csv.gz', 92650947)
    final_df = nf.get_all_network_statistics(g)
    final_df.to_csv('network_stats.csv', index = False)


if __name__ == '__main__':
    main()
