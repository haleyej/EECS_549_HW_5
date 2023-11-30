from document_preprocessor import SplitTokenizer, RegexTokenizer, SpaCyTokenizer
from indexing import Indexer, IndexType
from ranker import Ranker, WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF
import unittest
import shutil
import os

@unittest.skip('Broken tests')
class Test_SplitTokenizer_BasicInvertedIndex(unittest.TestCase):
    def convertFiles(self):
        if not os.path.isfile('multi_word_expressions.txt'):
            shutil.copy('multi_word_expressions_performance.txt', 'multi_word_expressions.txt')
        if not os.path.isfile('stopwords.txt'):
            shutil.copy('stopwords_performance.txt', 'stopwords.txt')

    def setUp(self):
            self.index_name = 'performance_index'
            self.index_type = IndexType.InvertedIndex
            self.preprocessor = SplitTokenizer
            self.convertFiles()
    
    def tearDown(self) -> None:
        os.remove('stopwords.txt')
        os.remove('multi_word_expressions.txt')

    def test_create_index_and_search_multiple_times(self):
        preprocessor = SplitTokenizer(file_path='multi_word_expressions.txt')
        LOOP_NUM = 15
        index = Indexer.create_index(
            self.index_name, IndexType.InvertedIndex, './performance_data.jsonl', preprocessor, True, 4)
        for _ in range(LOOP_NUM):
            ranker1 = Ranker(index, preprocessor, True, WordCountCosineSimilarity)
            ranker2 = Ranker(index, preprocessor, True, DirichletLM)
            ranker3 = Ranker(index, preprocessor, True, BM25)
            ranker4 = Ranker(index, preprocessor, True, PivotedNormalization)
            ranker5 = Ranker(index, preprocessor, True, TF_IDF)

            query = 'united nations and the korean war'

            ranker1.query(query)
            dlm = ranker2.query(query)
            bm25 =ranker3.query(query)
            ranker4.query(query)
            ranker5.query(query)
        shutil.rmtree('performance_index')
        self.assertEqual(dlm[0]['docid'], 60350145)
        self.assertEqual(bm25[0]['docid'], 60350145)

class Test_RegexTokenizer_BasicInvertedIndex(Test_SplitTokenizer_BasicInvertedIndex):
    def setUp(self):
        self.index_name = 'performance_index'
        self.index_type = IndexType.InvertedIndex
        self.preprocessor = RegexTokenizer
        self.convertFiles()

class Test_SpacyTokenizer_BasicInvertedIndex(Test_SplitTokenizer_BasicInvertedIndex):
    def setUp(self):
        self.index_name = 'performance_index'
        self.index_type = IndexType.InvertedIndex
        self.preprocessor = SpaCyTokenizer
        self.convertFiles()

class Test_SplitTokenizer_PositionalInvertedIndex(Test_SplitTokenizer_BasicInvertedIndex):
    def setUp(self):
            self.index_name = 'performance_index'
            self.index_type = IndexType.PositionalIndex
            self.preprocessor = SplitTokenizer
            self.convertFiles()

class Test_RegexTokenizer_PositionalInvertedIndex(Test_SplitTokenizer_BasicInvertedIndex):
    def setUp(self):
            self.index_name = 'performance_index'
            self.index_type = IndexType.PositionalIndex
            self.preprocessor = RegexTokenizer
            self.convertFiles()

class Test_SpacyTokenizer_PositionalInvertedIndex(Test_SplitTokenizer_BasicInvertedIndex):
    def setUp(self):
            self.index_name = 'performance_index'
            self.index_type = IndexType.PositionalIndex
            self.preprocessor = SpaCyTokenizer
            self.convertFiles()

if __name__ == '__main__':
    unittest.main()
