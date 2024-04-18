#
# For pre-processing data, we will use the NLTK library.
# based on the tutorial at https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
# The NLTK library is a powerful tool for text analytics. 
# It contains text processing libraries for tokenization, parsing, classification, stemming, tagging and semantic reasoning. 
# NLTK is largely used for teaching and research. It is accompanied by a number of corpora as well as a 
# GUI program called the NLTK downloader which makes it easy to download the resources.
# NLTK reference: https://arxiv.org/abs/2206.14774
#
#
# the training corpora are the following:
# - twitter_samples: 5000 positive and 5000 negative tweets
# - movie_reviews: 2000 positive and 2000 negative movie reviews
# - opinion_lexicon: positive and negative opinion words
# - sentiwordnet: sentiment scores for 155,000 words
# - wordnet: 155,287 words and 117,659 synonym sets
# - product_reviews_1: 400,000 positive and negative reviews
# - product_reviews_2: 400,000 positive and negative reviews
# - opinion_lexicon: 4,787 positive and 4,787 negative opinion words
#
# the classifiers are the following:
# - NaiveBayes
# - SVM
# - DecisionTree
# - RandomForest
# - LogisticRegression
# - KNN
# - GradientBoosting
# - NeuralNetwork
#
# Note that spaCy is used to provide named entity recognition and part of speech tagging.
# to install spacy, run the following command:
# pip install spacy
# python -m spacy download en_core_web_sm

import os
import pandas as pd
import re
import nltk
cwd = os.getcwd()
nltk.data.path.append(cwd + '/nltk_data')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import sklearn
import zipfile
import spacy

red = '\033[91m'
reset = '\033[0m'


nlp = spacy.load('en_core_web_sm')


def nltk_options():
    options = {'tweet_tokenizer': True, 'remove_stopwords': True, 'stem_words': True, 'lemmatize_words': False, 'nltk_dataset': "wordnet", 'nltk_classifiers': "NaiveBayes"}
    return options

def filter_directory_list( directory_list ):
    filtered_list = []
    for item in directory_list:
        if not item.endswith('.zip'):
            filtered_list.append(item)
    return filtered_list

# main nltk toolkit class

class NLTKToolkit:        
        nltk_data_dir = ""
        def __init__(self, root_dir, sg = None,  ): 
            self.df = None
            self.root_dir = root_dir
            self.nltk_data_dir = os.path.join(self.root_dir, 'nltk_data')
            self.text_column = "tweet"
            self.working_column = ""
            self.use_tweet_tokenizer = True
            self.nltk_dataset = None
            self.nltk_classifier_option = 'NaiveBayes'
            self.corpus = 'twitter_samples'
            self.nltk_classifier = NLTKClassifier( self.nltk_data_dir, self.text_column, self.use_tweet_tokenizer, self.nltk_classifier_option )
            self.trained_classifier = None
            self.nltk_classifier.root_dir = self.root_dir
            self.sg = sg

        def set_nltk_options(self, options):
            self.options = options
            self.remove_stopwords_option = options['remove_stopwords']
            self.stem_words_option = options['stem_words']
            self.lemmatize_words_option = options['lemmatize_words']

            if 'NaiveBayes' in options['nltk_classifiers']:
                self.nltk_classifier_option = 'NaiveBayes'
            elif 'SVM' in options['nltk_classifiers']:
                self.nltk_classifier_option = 'SVM'
            elif 'DecisionTree' in options['nltk_classifiers']:
                self.nltk_classifier_option = 'DecisionTree'
            elif 'RandomForest' in options['nltk_classifiers']:
                self.nltk_classifier_option = 'RandomForest'
            elif 'LogisticRegression' in options['nltk_classifiers']:
                self.nltk_classifier_option = 'LogisticRegression'
            elif 'KNN' in options['nltk_classifiers']:
                self.nltk_classifier_option = 'KNN'
            elif 'GradientBoosting' in options['nltk_classifiers']:
                self.nltk_classifier_option = 'GradientBoosting'
            elif 'NeuralNetwork' in options['nltk_classifiers']:
                self.nltk_classifier_option = 'NeuralNetwork'
            else:
                print("Invalid classifier option")

            if 'twitter_samples' in options['nltk_corpora']:
                self.corpus = 'twitter_samples'
            elif 'movie_reviews' in options['nltk_corpora']:
                self.corpus = 'movie_reviews'
            elif 'product_reviews_1' in options['nltk_corpora']:
                self.corpus = 'product_reviews_1'
            elif 'product_reviews_2' in options['nltk_corpora']:
                self.corpus = 'product_reviews_2'
            elif 'opinion_lexicon' in options['nltk_corpora']:
                self.corpus = 'opinion_lexicon'
            elif 'sentiwordnet' in options['nltk_corpora']:
                self.corpus = 'sentiwordnet'
            elif 'wordnet' in options['nltk_corpora']:
                self.corpus = 'wordnet'
            elif 'aclImdb' in options['nltk_corpora']:
                self.corpus = 'aclImdb'
            elif 'all' in options['nltk_corpora']:
                self.corpus = 'all'
            else:
                print("Invalid corpus option")
            self.nltk_classifier = NLTKClassifier( self.nltk_data_dir, self.text_column, self.use_tweet_tokenizer, self.nltk_classifier_option )
            return self.options
        
        def set_dataframe(self, df, text_column='tweet'):
            self.df = df
            self.working_column = text_column

        def run_model(self):
            if 'NaiveBayes' in self.nltk_classifier_option:                
                print("\nTraining Naive Bayes Classifier")
                self.trained_classifier = self.train_naive_bayes_classifier()
                print("\nRunning Naive Bayes Classifier on the column: ", self.working_column)
                self.df['sentiment'] = self.df[self.working_column].apply(lambda words: self.trained_classifier.classify({word: True for word in words}))
            elif self.nltk_classifier_option == 'SVM':
                print("\nTraining SVM Classifier")
                self.trained_classifier = self.train_svm_classifier()
                print("\nRunning SVM Classifier")
                self.df['sentiment'] = self.df[self.working_column].apply(lambda words: self.trained_classifier.classify({word: True for word in words}))
            elif self.nltk_classifier_option == 'DecisionTree':
                print("\nTraining Decision Tree Classifier")
                self.trained_classifier = self.train_decision_tree_classifier()
                print("\nRunning Decision Tree Classifier")
                self.df['sentiment'] = self.df[self.working_column].apply(lambda words: self.trained_classifier.classify({word: True for word in words}))
            elif self.nltk_classifier_option == 'RandomForest':
                print("\nTraining Random Forest Classifier")
                self.trained_classifier = self.train_random_forest_classifier()
                print("\nRunning Random Forest Classifier")
                self.df['sentiment'] = self.df[self.working_column].apply(lambda words: self.trained_classifier.predict({word: True for word in words}))
            elif self.nltk_classifier_option == 'LogisticRegression':
                self.trained_classifier = self.train_logistic_regression_classifier()
                print("\nRunning Logistic Regression Classifier")
                self.df['sentiment'] = self.df[self.working_column].apply(lambda words: self.trained_classifier.classify({word: True for word in words}))
            elif self.nltk_classifier_option == 'KNN':
                self.trained_classifier = self.train_knn_classifier()
                print("\nRunning KNN Classifier")
                self.df['sentiment'] = self.df[self.working_column].apply(lambda words: self.trained_classifier.classify({word: True for word in words}))
            elif self.nltk_classifier_option == 'GradientBoosting':
                self.trained_classifier = self.train_gradient_boosting_classifier()
                print("\nRunning Gradient Boosting Classifier")
                self.df['sentiment'] = self.df[self.working_column].apply(lambda words: self.trained_classifier.classify({word: True for word in words}))
            elif self.nltk_classifier_option == 'NeuralNetwork':
                self.trained_classifier = self.train_neural_network_classifier()
                print("\nRunning Neural Network Classifier")
                self.df['sentiment'] = self.df[self.working_column].apply(lambda words: self.trained_classifier.classify({word: True for word in words}))
            else:
                print("\nInvalid classifier option")

            self.df['named_entities'] = self.df[self.working_column].apply(self.get_named_entities)


            
            return self.df, self.working_column
        
        def get_named_entities(self, text):
            text = ' '.join(text)
            doc = nlp(text)
            named_entities = [(ent.text, ent.label_) for ent in doc.ents]
            return named_entities
        
        def split_featureset(self, featureset):
            from sklearn.model_selection import train_test_split
            training_set, testing_set = train_test_split(featureset, test_size=0.2, random_state=42)
            return training_set, testing_set

        def train_naive_bayes_classifier(self):
            from nltk.corpus import twitter_samples
            from nltk.classify import NaiveBayesClassifier
            from nltk.classify.util import accuracy
            
            positive_tweets = twitter_samples.strings('positive_tweets.json')
            negative_tweets = twitter_samples.strings('negative_tweets.json')

            positive_tweets = [(word_tokenize(tweet), 'positive') for tweet in positive_tweets]
            negative_tweets = [(word_tokenize(tweet), 'negative') for tweet in negative_tweets]

            tweets = positive_tweets + negative_tweets

            featureset = [(dict([token, True] for token in tweet), sentiment) for tweet, sentiment in tweets]

            training_set, testing_set = self.split_featureset(featureset)

            # create a Naive Bayes classifier
            classifier = NaiveBayesClassifier.train(training_set)
            acc = accuracy(classifier, testing_set)

            print(f"Naive Bayes Accuracy with {self.corpus} training set: {acc}")

            return classifier
        
        def train_svm_classifier(self):
            from nltk.classify import SklearnClassifier
            from sklearn.svm import SVC
            from nltk.classify.util import accuracy

            classifier = SklearnClassifier(SVC(), sparse=True)

            featureset = self.nltk_classifier.get_featureset(self.corpus)
            training_set, testing_set = self.split_featureset(featureset)
            classifier.train(training_set)
            acc = accuracy(classifier, testing_set)

            print(f"SVM Accuracy with {self.corpus} training set: {acc}")

            return classifier
        
        def train_decision_tree_classifier(self):
            from nltk.classify import DecisionTreeClassifier
            from nltk.classify.util import accuracy
            featureset = self.nltk_classifier.get_featureset(self.corpus)

            for features, label in featureset[:5]:
                print(f"Features: {features}, Label: {label}")

            training_set, testing_set = self.split_featureset(featureset)

            classifier = DecisionTreeClassifier.train(training_set)

            acc = accuracy(classifier, testing_set)

            print(f"Decision Tree Accuracy with {self.corpus} training set: {acc}")
            return classifier

        def train_random_forest_classifier(self):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.metrics import accuracy_score

            featureset = self.nltk_classifier.get_featureset(self.corpus)

            features, labels = zip(*featureset)
            vectorizer = DictVectorizer()
            X = vectorizer.fit_transform(features)
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

            classifier = RandomForestClassifier(n_estimators=10)
            classifier.fit(X_train, y_train)

            y_pred_test = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred_test)

            print(f"Random Forest Accuracy with {self.corpus} training set: {acc}")
            return classifier

        def train_logistic_regression_classifier(self):
            from nltk.classify import MaxentClassifier
            from nltk.classify.util import accuracy
            featureset = self.nltk_classifier.get_featureset(self.corpus)

            training_set, testing_set = self.split_featureset(featureset)

            classifier = MaxentClassifier.train(training_set)

            acc = accuracy(classifier, testing_set)

            print(f"Logistic Regression Accuracy with {self.corpus} training set: {acc}")
            return classifier

        def train_knn_classifier(self):
            from nltk.classify import SklearnClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from nltk.classify.util import accuracy
            featureset = self.nltk_classifier.get_featureset(self.corpus)

            training_set, testing_set = self.split_featureset(featureset)

            classifier = SklearnClassifier(KNeighborsClassifier(), sparse=True)

            classifier.train(training_set)

            acc = accuracy(classifier, testing_set)

            print(f"KNN Accuracy with {self.corpus} training set: {acc}")
            return classifier

        def train_gradient_boosting_classifier(self):
            from nltk.classify import SklearnClassifier
            from sklearn.ensemble import GradientBoostingClassifier
            from nltk.classify.util import accuracy
            featureset = self.nltk_classifier.get_featureset(self.corpus)

            training_set, testing_set = self.split_featureset(featureset)

            classifier = SklearnClassifier(GradientBoostingClassifier(), sparse=True)

            classifier.train(training_set)

            acc = accuracy(classifier, testing_set)

            print(f"Gradient Boosting Accuracy with {self.corpus} training set: {acc}")
            return classifier

        def train_neural_network_classifier(self):
            from nltk.classify import SklearnClassifier
            from sklearn.neural_network import MLPClassifier
            from nltk.classify.util import accuracy
            featureset = self.nltk_classifier.get_featureset(self.corpus)

            training_set, testing_set = self.split_featureset(featureset)

            classifier = SklearnClassifier(MLPClassifier(), sparse=True)

            classifier.train(training_set)

            acc = accuracy(classifier, testing_set)

            print(f"Neural Network Accuracy with {self.corpus} training set: {acc}")
            return classifier  


            

        def display_nltk_data(self):
            nltk_data = self.nltk_classifier.get_nltk_data()
            print("Local NLTK Data: \n")
            for title, data in nltk_data:
                print(f"{title}: {data}")
        
        def clean_df(self):
            from datasets.datasets import clean_and_parse_df
            if 'cleaned_text' not in self.df.columns:
                print(self.df.head()) 
                self.df = clean_and_parse_df(self.df, self.text_column)
            self.working_column = 'cleaned_text'
            return self.df, 'cleaned_text'
    
        def remove_stopwords(self, text):
            tokenized_text = ''
            stop_words = stopwords.words('english')
            # tokenization
            if self.use_tweet_tokenizer:
                tokenized_text = nltk.tokenize.TweetTokenizer().tokenize(text)
            else:
                tokenized_text = word_tokenize(text)
            # remove stopwords
            text = [word for word in tokenized_text if word not in stop_words]
            return text
        
        def remove_stopwords_df(self):
            self.df['cleaned_text_no_stopwords'] = self.df['cleaned_text'].apply(self.remove_stopwords)
            self.working_column = 'cleaned_text_no_stopwords'
            return self.df, 'cleaned_text_no_stopwords'
    
        def stem_words(self, text):
            stemmer = PorterStemmer()
            text = [stemmer.stem(word) for word in text]
            return text

        def stem_words_df(self):
            self.df['cleaned_text_no_stopwords_stemmed'] = self.df['cleaned_text_no_stopwords'].apply(self.stem_words)
            return self.df
        
        def lemmatize_words(self, text):
            lemmatizer = WordNetLemmatizer()
            text = [lemmatizer.lemmatize(word) for word in text]
            return text

        def lemmatize_words_df(self):
            self.df[self.working_column] = self.df[self.working_column].apply(self.lemmatize_words)
            return self.df
        
        def tokenize_text(self, column_name_to_tokenize):
            # tokenize the text
            if self.use_tweet_tokenizer:
                tokenized_text = nltk.tokenize.TweetTokenizer().tokenize(self.df[column_name_to_tokenize])
            else:
                tokenized_text = word_tokenize(self.df[self.column_name_to_tokenize])
            print(tokenized_text)
            return tokenized_text

        def tokenize_text_df(self):
            self.df['cleaned_text_no_stopwords_tokenized'] = self.df['cleaned_text_no_stopwords'].apply(self.tokenize_text)
            return self.df
        
        # the standard vader lexicon used for standard sentiment analysis
        
        def run_nltk_sentiment_analysis(self):
            if not os.path.isdir(self.nltk_data_dir + '/sentiment'):
                self.nltk_classifier.download_nltk_dataset('vader_lexicon')
            if os.path.isfile(self.nltk_data_dir + '/sentiment/vader_lexicon.zip'):
                with zipfile.ZipFile(self.nltk_data_dir + '/sentiment/vader_lexicon.zip', 'r') as zip_ref:
                    zip_ref.extractall(self.nltk_data_dir + '/sentiment')
            sid = SentimentIntensityAnalyzer( 'sentiment/vader_lexicon/vader_lexicon.txt' )
            self.df['sentiment'] = self.df[self.working_column].apply(lambda x: sid.polarity_scores(' '.join(x)))
            return self.df, 'sentiment'

class NLTKClassifier:
    
    def __init__(self, data_dir, df=None, text_column='text', use_tweet_tokenizer=False, classifier=None):
        self.df = df
        self.text_column = text_column
        self.use_tweet_tokenizer = use_tweet_tokenizer
        self.dataset = None
        self.classifier = classifier
        self.data_dir = data_dir
        self.root_dir = ''
        if self.check_nltk_data():
            self.corpora = filter_directory_list( os.listdir(os.path.join(data_dir, 'corpora')))
            self.models = filter_directory_list( os.listdir(os.path.join(data_dir,'models')))
            self.taggers = filter_directory_list( os.listdir(os.path.join(data_dir, 'taggers')))
            self.tokenizers = filter_directory_list( os.listdir(os.path.join(data_dir, 'tokenizers')))
            self.stemmers = filter_directory_list( os.listdir(os.path.join(data_dir, 'stemmers')))
            self.chunkers = filter_directory_list( os.listdir(os.path.join(data_dir, 'chunkers')))
            self.grammars = filter_directory_list( os.listdir(os.path.join(data_dir, 'grammars')))
            self.misc = filter_directory_list( os.listdir(os.path.join(data_dir, 'misc')))
            self.sentiment = filter_directory_list( os.listdir(os.path.join(data_dir, 'sentiment')))

    def get_nltk_data(self):
        data = [ ('corpora', self.corpora),
                 ('models', self.models),
                 ('taggers', self.taggers), ('tokenizers',self.tokenizers), 
                 ('stemmers',self.stemmers ), ('chunkers', self.chunkers), 
                 ('grammars', self.grammars), ('misc', self.misc), ('sentiment', self.sentiment) ]
        return data

    def check_nltk_data(self):
        cwd = os.getcwd()
        if not os.path.isdir(cwd + '/nltk_data'):
            return False
        else:
            return True
        
    def set_classifier(self, classifier):
        self.classifier = classifier
        return self.classifier
    
    def set_corpus(self, corpus):
        self.dataset = corpus
        return self.dataset
    
    def set_model(self, model):
        self.dataset = model
        return self.dataset
    
    def set_tagger(self, tagger):
        self.dataset = tagger
        return self.dataset
    
    def set_tokenizer(self, tokenizer):
        self.dataset = tokenizer
        return self.dataset
    
    def set_stemmer(self, stemmer):
        self.dataset = stemmer
        return self.dataset
    
    def set_chunker(self, chunker):
        self.dataset = chunker
        return self.dataset
    
    def set_grammar(self, grammar):
        self.dataset = grammar
        return self.dataset
    
    def set_misc(self, misc):
        self.dataset = misc
        return self.dataset
    
    def set_sentiment(self, sentiment):
        self.dataset = sentiment
        return self.dataset
        
        
    def download_nltk_dataset(self, dataset_name="wordnet"):
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd, 'nltk_data', dataset_name)):
            nltk_data_dir = os.path.join(cwd, 'nltk_data')
            nltk.download(info_or_id=dataset_name, download_dir=nltk_data_dir)

    def load_nltk_dataset(self, dataset_name):
        dataset = nltk.corpus.load(dataset_name)
        from nltk.corpus import wordnet as dataset
        return dataset
    
    def normalize_sentences(self, tokens):
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        normalized_sentences = []
        for token in tokens:
            stemmed_token = stemmer.stem(token)
            lemmatized_token = lemmatizer.lemmatize(stemmed_token)
            normalized_sentences.append(lemmatized_token)
        return normalized_sentences
    

    # this is a helper function for strings that are not tokenized
    def document_features(document, word_features):
                document_words = set(document)
                features = {}
                for word in word_features:
                    features['contains({})'.format(word)] = (word in document_words)
                return features
    
    # a helper function for words
    def get_word_features(words):
        useful_words = [word for word in words if word not in stopwords.words("english")]
        my_dict = dict([(word, True) for word in useful_words])
        return my_dict
    
    def create_wordnet_features(classifier, words):
        from nltk.corpus import wordnet as wn

        useful_words = [word for word in words if word not in nltk.corpus.stopwords.words("english")]
        synonyms = []
        for word in useful_words:
            for syn in wn.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
        my_dict = dict([(word, True) for word in synonyms])
        return my_dict
    
    def get_featureset(self, corpus = 'twitter_samples'):
        if corpus == 'all':
            all_corpora = ['twitter_samples', 'movie_reviews', 'opinion_lexicon', 'sentiwordnet'] # not aclImdb
            all_featuresets = []
            for c in all_corpora:
                all_featuresets.extend(self.get_named_featureset(c))
            return all_featuresets
        else:
            return self.get_named_featureset(corpus)
    
    def get_named_featureset( self, corpus = 'twitter_samples' ):
        feature_sets = []
        if corpus == 'twitter_samples':
            # this is working
            from nltk.corpus import twitter_samples
            positive_tweets = twitter_samples.strings('positive_tweets.json')
            negative_tweets = twitter_samples.strings('negative_tweets.json')

            positive_tweets = [(word_tokenize(tweet), 'positive') for tweet in positive_tweets]
            negative_tweets = [(word_tokenize(tweet), 'negative') for tweet in negative_tweets]

            tweets = positive_tweets + negative_tweets

            feature_sets = [(dict([token, True] for token in tweet), sentiment) for tweet, sentiment in tweets]

        elif corpus == 'movie_reviews':
            # this is working
            from nltk.corpus import movie_reviews
            from nltk.classify.util import apply_features

            documents = [(str(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

            import random
            random.shuffle(documents)

            reviews = [(word_tokenize(review), sentiment) for review, sentiment in documents]

            featureset = [(dict([token, True] for token in review), sentiment) for review, sentiment in reviews]

            feature_sets.extend(featureset)
        elif corpus == 'product_reviews_1':
            # this is working
            from nltk.corpus import product_reviews_1

            reviews = product_reviews_1.reviews()

            feature_sets = []

            for review in reviews:
                features = {}
                sentiments = []
                for feature, sentiment in review.features():
                    features[feature] = True
                    sentiments.append(sentiment)
                sentiment = 'positive' if sentiments and  max(sentiments, key=sentiments.count) == '+1' or max(sentiments, key=sentiments.count) == '+2' or max(sentiments, key=sentiments.count) == '+3' else 'negative'
                feature_sets.append((features, sentiment))            


        elif corpus == 'product_reviews_2':
            # this is working
            from nltk.corpus import product_reviews_2
            from nltk.classify.util import apply_features

            nltk.download('product_reviews_2')

            reviews = product_reviews_2.fileids()

            for fileid in reviews:
                review = product_reviews_2.raw(fileid)
                features = {
                    "has_positive_opinion": "[+1]" in review,
                    "has_negative_opinion": "[-1]" in review,
                    "has_positive_opinion": "[+2]" in review,
                    "has_negative_opinion": "[-2]" in review,
                    "has_positive_opinion": "[+3]" in review,
                    "has_negative_opinion": "[-3]" in review,
                    "has_unknown_feature": "[u]" in review,
                }
                if features["has_positive_opinion"]:
                    sentiment = "positive"
                elif features["has_negative_opinion"]:
                    sentiment = "negative"
                else:
                    sentiment = "neutral" 
                feature_sets.extend((features, sentiment))

        elif corpus == 'opinion_lexicon':
            # this is working
            from nltk.corpus import opinion_lexicon

            positive_words = opinion_lexicon.positive()
            print(positive_words)
            negative_words = opinion_lexicon.negative()

            positive_features = [(self.create_wordnet_features(word), 'positive') for word in positive_words]
            negative_features = [(self.create_wordnet_features(word), 'negative') for word in negative_words]

            feature_sets = positive_features + negative_features
        elif corpus == 'sentiwordnet':
            # this is working
            from nltk.corpus import sentiwordnet as swn

            all_synsets = list(swn.all_senti_synsets())

            sentiment_scores = {}

            for synset in all_synsets:
                pos_score = synset.pos_score()
                neg_score = synset.neg_score()
                sentiment_score = pos_score - neg_score

                for lemma in synset.synset.lemmas():
                    sentiment_scores[lemma.name()] = sentiment_score

            from nltk.corpus import twitter_samples
            tweets = twitter_samples.strings('positive_tweets.json') + twitter_samples.strings('negative_tweets.json')
            tokenized_tweets = [word_tokenize(tweet) for tweet in tweets]

            sentiment_scores_list = [sum(sentiment_scores.get(word, 0) for word in tweet) for tweet in tokenized_tweets]

            feature_sets = [(dict([token, True] for token in tweet), sentiment) for tweet, sentiment in zip(tokenized_tweets, sentiment_scores_list)] 

        elif corpus == 'wordnet':
            from nltk.corpus import wordnet as wn
            from nltk.corpus import opinion_lexicon

            positive_words = opinion_lexicon.positive()
            negative_words = opinion_lexicon.negative()

            positive_features = [(self.create_wordnet_features(word), 'positive') for word in positive_words]
            negative_features = [(self.create_wordnet_features(word), 'negative') for word in negative_words]

            feature_sets = negative_features + positive_features
        elif corpus == 'aclImdb':
            aclImdb_dir = os.path.join(self.root_dir, 'TwitterArchive', 'TwitterArchive', 'aclImdb')

        # Define subdirectories for positive and negative reviews
            pos_dir = os.path.join(aclImdb_dir, 'train', 'pos')
            neg_dir = os.path.join(aclImdb_dir, 'train', 'neg')

            for filename in os.listdir(pos_dir):
                if filename.endswith('.txt'):
                    with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as file:
                        review_text = file.read()
                        feature_sets.append((self.create_wordnet_features(word), 'positive') for word in review_text)  # Positive label

            for filename in os.listdir(neg_dir):
                if filename.endswith('.txt'):
                    with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as file:
                        review_text = file.read()
                        feature_sets.append((self.create_wordnet_features(word), 'negative') for word in review_text )  # Negative label
        print("\n===============================================\n")
        return feature_sets
    
    
    

    
    
    

    


