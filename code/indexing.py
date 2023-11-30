from enum import Enum
import json
import os
from tqdm import tqdm
from collections import Counter, defaultdict
import shelve
from document_preprocessor import Tokenizer
import gzip
import numpy as np
import shutil


class IndexType(Enum):
    # The three types of index currently supported are InvertedIndex, PositionalIndex and OnDiskInvertedIndex
    InvertedIndex = 'BasicInvertedIndex'
    # NOTE: You don't need to support the following three
    PositionalIndex = 'PositionalIndex'
    OnDiskInvertedIndex = 'OnDiskInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    def __init__(self) -> None:
        """
        The base interface representing the data structure for all index classes.
        The functions are meant to be implemented in the actual index classes and not as part of this interface.
        """
        self.statistics = defaultdict(Counter)  # Central statistics of the index
        self.index = {}  # Index
        self.document_metadata = {}  # Metadata like length, number of unique tokens of the documents
        self.term_metadata = {}

    # NOTE: The following functions have to be implemented in the three inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        # TODO: Implement this to remove a document from the entire index and statistics
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Adds a document to the index and updates the index's metadata on the basis of this
        document's addition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        # TODO: Implement this to add documents to the index
        raise NotImplementedError

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.
        
        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in 
            the document
        """
        # TODO: Implement this to fetch a term's postings from the index
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        # TODO: Implement to fetch a particular document stored in metadata
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "count": How many times this term appeared in the corpus as a whole

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        # TODO: Implement to fetch a particular term stored in metadata
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary mapping statistical properties (named as strings) about the index to their values.  
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens), 
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
              A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        # TODO: Calculate statistics like 'unique_token_count', 'total_token_count',
        #       'number_of_documents', 'mean_document_length' and any other relevant central statistic
        raise NotImplementedError

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        # TODO: Save the index files to disk
        raise NotImplementedError

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save().

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        # TODO: Load the index files from disk to a Python object
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        self.docs_to_words = {}
        self.vocabulary = set()

    def remove_doc(self, docid: int) -> None:
        for token, docs in list(self.index.items()):
            for i, doc in enumerate(docs):
                if int(doc[0]) == int(docid):
                    del docs[i]
                    break
                if len(docs) == 0:
                    del self.index[token]
                    self.vocabulary.remove(token)
                    del self.term_metadata[token]

        del self.document_metadata[docid]

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        '''
        Adds a document to the index and updates the index's metadata on the basis of this
        document's addition (e.g., collection size, average document length, etc.)

        Arguments:
            docid [int]: the identifier of the document

            tokens list[str]: the tokens of the document. Tokens that should not be indexed will have 
            been replaced with None in this list. The length of the list should be equal to the number
            of tokens prior to any token removal.
        '''
        self.docs_to_words[docid] = tokens[:500]
        counts = Counter(tokens)
        for token in set(tokens):
            if token == None:
                continue
            self.vocabulary.add(token)
            term_count = counts[token]
            if token in self.term_metadata.keys():
                if self.index[token] == None:
                    continue
                new = (docid, term_count)
                current = self.index[token]
                # idx = bisect.bisect_left(current, new)
                # current.insert(idx, new)
                current.append(new)
                self.index[token] = current

                self.term_metadata[token]['count']  = self.term_metadata[token].get('count', 0) + term_count 
                self.term_metadata[token]['n_docs'] = self.term_metadata[token].get('n_docs', 0) + 1
            else:
                self.index[token] = [(docid, term_count)]
                self.term_metadata[token] = {}
                self.term_metadata[token]['count']  = term_count 
                self.term_metadata[token]['n_docs'] =  1

        # doc metadata
        num_tokens = len(tokens)
        unique_tokens = set(tokens)
        unique_tokens.discard(None)

        self.document_metadata[docid] = {'num_unique_tokens': len(unique_tokens), 'total_token_count': num_tokens}

        length = len(tokens)
        if length == 0:
            self.document_metadata[docid] = {'length': 0, 'unique_tokens': 0}
            return
         
    def get_postings(self, term: str) -> list[tuple[int, str]]:
        '''
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.
        
        Arguments:
            term [str]: the term to be searched for

        Returns:
            list[tuple[int,str]] : A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in 
            the document.
        '''
        return self.index.get(term, None)

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        '''
        For the given document id, returns a dictionary with metadata about that document. Metadata
        should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)             
        '''
        return self.document_metadata.get(doc_id, {})

    def get_term_metadata(self, term: str) -> dict[str, int]:
        '''
        For the given term, returns a dictionary with metadata about that term in the index. Metadata
        should include keys such as the following:
            "count": How many times this term appeared in the corpus as a whole.          
        '''        
        return self.term_metadata.get(term, {})

    def get_statistics(self) -> dict[str, int]:
        '''
        Returns a dictionary mapping statistical properties (named as strings) about the index to their values.  
        Keys should include at least the following:

            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens), 
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)                
        '''
        doc_metadata = self.document_metadata

        tokens_per_doc = [doc_metadata[k].get('total_token_count', 0) for k in list(doc_metadata.keys())]
        all_tokens = [doc_metadata[k].get('total_token_count', ) for k in list(doc_metadata.keys())]
        stats = {
                    'number_of_documents': len(list(doc_metadata.keys())), 
                    'mean_document_length': np.nan_to_num(np.mean(tokens_per_doc)), 
                    'total_token_count': np.sum(all_tokens),
                    'unique_token_count': len(self.vocabulary) 
                }
        return stats

    def save(self, index_directory_name) -> None:
        '''
        Saves the state of this index to the provided directory. The save state should include the
        inverted index as well as any meta data need to load this index back from disk
        '''
        if os.path.exists(index_directory_name):
            shutil.rmtree(index_directory_name)
        os.mkdir(index_directory_name)
        with open(os.path.join(index_directory_name, 'BasicIndex' + ".json"), "w") as f:
            f.write(json.dumps(self.index))

    def load(self, index_directory_name) -> None:
        '''
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save()
        '''
        if os.path.exists(os.path.join(index_directory_name, 'BasicIndex' + ".json")):
            with open(os.path.join(index_directory_name, 'BasicIndex' + ".json"), "r") as f:
                self.index = json.loads(f.read())

    
  
class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self, index_name) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__(index_name)
        self.statistics['index_type'] = 'PositionalInvertedIndex'
        # For example, you can initialize the index and statistics here:
        #   self.statistics['offset'] = [0]
        #   self.statistics['docmap'] = {}
        #   self.doc_id = 0
        #   self.postings_id = -1

    # TODO: Do nothing, unless you want to explore using a positional index for some cool features


class OnDiskInvertedIndex(BasicInvertedIndex):
    def __init__(self, shelve_filename) -> None:
        """
        This is an inverted index where the inverted index's keys (words) are kept in memory but the
        postings (list of documents) are on desk.
        The on-disk part is expected to be handled via a library.
        """
        super().__init__()
        self.shelve_filename = shelve_filename
        self.statistics['index_type'] = 'OnDiskInvertedIndex'
        # Ensure that the directory exists
        # self.index = shelve.open(self.shelve_filename, 'index')
        # self.statistics['docmap'] = {}
        # self.doc_id = 0

    # NOTE: Do nothing, unless you want to re-experience the pain of cross-platform compatibility :'(


class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    @staticmethod 
    def load_docs(doc: dict, text_key: str, document_preprocessor: Tokenizer, doc_augment_dict: dict[int, list[str]] | None = None):
        text = doc[text_key]
        docid = int(doc['docid'])
        if doc_augment_dict != None:
            augmentations = doc_augment_dict.get(docid)
            if augmentations != None: 
                for augmentation in augmentations: 
                    text += (" " + augmentation)
        tokens = document_preprocessor.tokenize(text)
        return (docid, tokens)

    @staticmethod 
    def add_to_index(doc_info:list, do_not_index:list[str], index:InvertedIndex):
        docid, tokens = doc_info
        if tokens == None or len(tokens) == 0:
            return index
        for i, token in enumerate(tokens):
            if token.lower() in do_not_index:
                tokens[i] = None
        index.add_doc(docid, tokens)
        return index

    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str, 
                     document_preprocessor: Tokenizer, stopwords: set[str], 
                     minimum_word_frequency: int, text_key="text",
                     max_docs=-1, doc_augment_dict: dict[int, list[str]] | None = None) -> InvertedIndex:
        '''
        The Index class' static function which is responsible for creating an inverted index

        Parameters:        

        index_type [IndexType]: This parameter tells you which type of index to create, e.g., a BasicInvertedIndex.

        dataset_path [str]: This is the file path to your dataset

        document_preprocessor: This is a class which has a 'tokenize' function which would read each document's text and return back a list of valid tokens.

        stopwords [set[str]]: The set of stopwords to remove during preprocessing or `None` if no stopword preprocessing is to be done.

        minimum_word_frequency [int]: This is also an optional configuration which sets the minimum word frequency of a particular token to be indexed. If the token does not appear in the document atleast for the set frequency, it will not be indexed. Setting a value of 0 will completely ignore the parameter.

        text_key [str]: the key in the JSON to use for loading the text. 

        max_docs [int]: The maximum number of documents to index. Documents are processed in the order they are seen

        '''        
        index = BasicInvertedIndex()
        
        i = 0
        if dataset_path.endswith(".gz"):
            with gzip.open(dataset_path, 'r') as f:
                doc = f.readline()
                while doc and (i < max_docs or max_docs == -1):
                    doc = json.loads(doc)
                    doc_info = __class__.load_docs(doc, text_key, document_preprocessor, doc_augment_dict)
                    index = __class__.add_to_index(doc_info, stopwords, index)
                    doc = f.readline()
                    i += 1
        elif dataset_path.endswith(".jsonl"):
            with open(dataset_path, 'r') as f:
                doc = f.readline()
                while doc and (i < max_docs or max_docs == -1):
                    doc = json.loads(doc)
                    doc_info = __class__.load_docs(doc, text_key, document_preprocessor, doc_augment_dict)
                    index = __class__.add_to_index(doc_info, stopwords, index)
                    doc = f.readline() 
                    i += 1

        if minimum_word_frequency > 1:
            metadata = index.term_metadata
            for term in tqdm(list(metadata.keys())):
                if metadata[term]['count'] < minimum_word_frequency:
                    postings = index.get_postings(term)
                    for posting in postings:
                        docid, _ = posting
                        index.document_metadata[docid]['num_unique_tokens'] = index.document_metadata[docid]['num_unique_tokens'] - 1
                    index.vocabulary.remove(term)
                    del index.index[term]
                    del index.term_metadata[term]
        return index