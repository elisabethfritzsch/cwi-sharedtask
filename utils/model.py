from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import re
import pickle
from nltk.corpus import wordnet as wn
import numpy as np
from utils.affixes import Affixes
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve



class Model(object):
    def __init__(self, language, is_baseline):
        self.language = language
        self.is_baseline = is_baseline
        self.lang_code = self.language[:3]
        
        # Create model
        if is_baseline:
             self.model = LogisticRegression()
        else:
            self.model = RandomForestClassifier(n_estimators=10, random_state = np.random.seed(0))
        
        # Get word and trigram frequency counts
        with open('utils/' + language + '_freqs.pkl', 'rb') as f:
                self.freqs = pickle.load(f)
        with open('utils/' + language + '_tf.pkl', 'rb') as tf:
                self.tri_freqs = pickle.load(tf)
                
                
        # Define average word length (from 'Multilingual and Cross-Lingual 
        # Complex Word Identification' (Yimam et. al, 2017)) and vowel clusters for syllable counts
        if language == 'english':
            self.avg_word_length = 5.3
            self.syll = re.compile('[aeiouy]+', re.IGNORECASE)
        else:  
            self.avg_word_length = 6.2
            self.syll = re.compile('a[uiy]|e[iy]|i[aeouáéóú]|o[iy]|u[aeio]|[aeiouáéíóú]', re.IGNORECASE)

        # Get list of Greek and Latin affixes
        aff = Affixes()
        self.affixes = aff.affixes
    
    # Train model
    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])
        self.model.fit(X, y)
        
    # Evaluate model on unseen data
    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))
        return self.model.predict(X)
     
    # Extract relevant features from a target word   
    def extract_features(self, word):
        
        # Get word length in characters and tokens
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        
        # Approximate syllables
        clusters = len(re.findall(self.syll, word, ))
        if clusters == 0:
            clusters = 1
        
        if self.is_baseline:
            return[len_chars, len_tokens, clusters]
        
        # Get word and trigram frequencies
        trigram_freq = self.tri_freqs[word]
        freq = self.freqs[word]

        # Check if word contains Latin or Greek affixes
        lat_or_greek = int(any(affix in word for affix in self.affixes))
        
        # Count number of senses and synonyms
        no_senses = len(wn.synsets(word, lang = self.lang_code))
        synonyms = set()
        for synset in wn.synsets(word, lang = self.lang_code):
            for lemma in synset.lemma_names(self.lang_code):
                synonyms.add(lemma)
        no_synonyms = len(synonyms)
        return [len_chars, len_tokens, freq, no_senses, no_synonyms, lat_or_greek, clusters, trigram_freq]
       
    # Print out exemplary decision tree. Only works is self.model=DecisionTreeClassifier()
    def get_decision_tree(self):
        with open('{}_tree.dot'.format(self.lang_code), "w") as f:
            f = tree.export_graphviz( self.model, out_file=f, 
                        feature_names = ['len_chars', 'len_tokens', 'clusters'], #replace with appropriate feature names
                         class_names=['simple','complex'],  
                         filled=True, rounded=True,  
                         special_characters=True)
            


    # Plot learning curve. Code from http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    
        plt.legend(loc="best")
        return plt
    
    def plot(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])
        title = "Learning Curves (Random forest, all features, {})".format(self.language)
        self.plot_learning_curve(self.model, title, X, y)
        plt.show()


