import os
import pickle
import numpy as np
import pandas as pd
import json
import re
from BM25 import BM25
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from nltk.tokenize import word_tokenize
from Pr import Pr

def preProcess(s):
    s = re.sub(r'[^A-Za-z]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    s = word_tokenize(s)
    return ' '.join(s)


class ManualIndexer:
    def __init__(self):
        self.crawled_folder = Path(os.path.abspath('')).parent / 'crawled/'
        self.stored_file = 'manual_indexer.pkl'
        if os.path.isfile(self.stored_file):
            with open(self.stored_file, 'rb') as f:
                cache_dict = pickle.load(f)
            self.__dict__.update(cache_dict)
        else:
            self.run_indexer()

    def run_indexer(self):
        documents = []
        self.pr = Pr(alpha=0.85)
        self.pr.pr_calc()
        for file in os.listdir(self.crawled_folder):
            if file.endswith(".txt"):
                j = json.load(open(os.path.join(self.crawled_folder, file)))
                j['id'] = j['url']
                j['pagerank'] = self.pr.pr_result.loc[j['id']].score
                print(j)
                documents.append(j)
        self.documents = pd.DataFrame.from_dict(documents)
        self.page_rank = self.documents['pagerank'].array
        tfidf_vectorizor = TfidfVectorizer(preprocessor=preProcess, stop_words=stopwords.words('english'))
        self.bm25 = BM25(tfidf_vectorizor)

        self.bm25.fit(self.documents.apply(lambda s: ' '.join(s[['title', 'text']]), axis=1))
        with open(self.stored_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def query(self, q):
        return_pr_list = self.page_rank
        return_score_list = self.bm25.transform(q)
        hit = (return_pr_list > 0).sum()
        rank = return_pr_list.argsort()[::-1][:hit]
        results = self.documents.iloc[rank].copy().reset_index(drop=True)
        results['score'] = return_pr_list[rank]

        # Combine BM25 scores with PageRank scores
        combined_scores = self.page_rank * return_score_list
        # Sort results based on combined scores
        rank = combined_scores.argsort()[::-1][:hit]
        results = self.documents.iloc[rank].copy().reset_index(drop=True)
        results['score'] = combined_scores[rank]
        return results


if __name__ == "__main__":
    indexer = ManualIndexer()
    indexer.run_indexer()
    results = indexer.query('camt')