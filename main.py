import collections

from nltk import word_tokenize
#from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
#from pprint import pprint
#import codecs
#from sklearn import feature_extraction 
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
#from sklearn.cluster import DBSCAN

import hdbscan

import numpy as np

#import warnings

#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

# from sklearn import metrics
import pandas as pd


#%%

# make matplotlib plot inline (Only in Ipython).
#warnings.filterwarnings('ignore')
#%matplotlib inline

import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

class Clustering:
    def __init__(self):
        punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
        self.custom_stopwords = text.ENGLISH_STOP_WORDS.union(punc)
        pass
    
     #%%
    def word_tokenizer(self,text):
        tokens = word_tokenize(text)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
        return tokens

    ### stopwords and pontuations 
    

    def loadCorpus(self):
        
        files_array = ['./data/linguistic.txt', 
                       './data/machinelearning.txt', 
                       './data/ciclism.txt', 
                       './data/religion.txt', 
                       './data/atlanticocean.txt']
        corpus = []
        
        for path in files_array:
            with open(path, 'r') as t:
                for line in t.readlines():
                    corpus.append(line.strip())

        return corpus           

    def transform(self):
        
        self.train_corpus = self.loadCorpus()

        self.tfidf_vect = TfidfVectorizer(tokenizer=self.word_tokenizer,
                                    stop_words=self.custom_stopwords,                            
                                    max_df=0.95,                            
                                    max_features=300,
                                    lowercase=True)

        train = self.tfidf_vect.fit_transform(self.train_corpus)
        
        self.train_array = train.toarray()
        print(self.train_array.shape) #, _train.get_feature_names())
        
    def fit(self):
        self.transform()
        
        clusters = hdbscan.HDBSCAN(min_cluster_size=21, prediction_data=True).fit(self.train_array)
        self.hdbscan = hdbscan
        self.clusters = clusters
        print(np.unique(self.clusters.labels_))

    def mountSentencesRank(self):
        rank = pd.DataFrame({'n_cluster': [], 'idx_word': [], 'word': [], 'tfidf': []})
        for c, i in enumerate(self.clusters.labels_):
            if i == -1:
                continue
            rank = rank.append({'n_cluster': i, 'idx_word': c, 'word': self.train_corpus[c], 'tfidf': self.train_array[c].mean()}, ignore_index=True)
        return rank

    def query(self, text):

        test = self.tfidf_vect.transform(text)
        test_array = test.toarray()

        query_labels, strengths = self.hdbscan.approximate_predict(self.clusters, test_array)
        label_pred = query_labels[0]
        
        print(label_pred)

        rank = self.mountSentencesRank()

        res = rank[rank['n_cluster'] == label_pred].sort_values('tfidf', ascending=False).head(3)

        most_frequent = res['word'].values
        print(len(most_frequent))
        return most_frequent


if __name__ == '__main__':
    
    model = Clustering()
    print("Training")
    model.fit()
    
    query_text = ['Machine learning explores the study and construction of algorithms that can learn from and make predictions on data – such algorithms overcome following strictly static program instructions by making data-driven predictions or decisions, through building a model from sample inputs.']
    
    print("Testing")
    
    most_frequent_sentences = model.query(query_text)
    
    for i, sentence in enumerate(most_frequent_sentences):
        print("*"*20)
        print(i, sentence)