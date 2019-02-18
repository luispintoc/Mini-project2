import os, re, json, math
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.corpus import stopwords, movie_reviews
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from pattern.en import sentiment
import warnings
import nltk
warnings.filterwarnings("ignore",category = FutureWarning)

#		*****Initialization*****

target = [] #len = 25000
reviews = [] #Shuffled training data, len = 25000
positive_words = []
negative_words = []
bing_liu_list = []
stopwords = []

#		********Read data*********

with open("train_data.json") as fp:
    train_data = json.load(fp)
i = 0
while i < 25000:
	target.append(train_data[i][0])
	reviews.append(train_data[i][1])
	i += 1

with open('positive-words.txt') as f:
    positive_words = f.read().splitlines()
with open('negative-words.txt') as f:
    negative_words = f.read().splitlines()
with open('stopwords.txt') as f:
    stopwords = f.read().splitlines()

bing_liu_list = list(zip(positive_words,negative_words))
words_for_movies = list(movie_reviews.words())
vocab = list(set(bing_liu_list + words_for_movies))

#		********Preprocessing********

delete = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
replace_with_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def compile(reviews):
	reviews = [delete.sub("",line.lower()) for line in reviews]
	reviews = [replace_with_space.sub(" ",line) for line in reviews]
	return reviews

def get_sentiment(x):
    return np.array([sentiment(t)[0] for t in x]).reshape(-1, 1)

def get_sentiment2(x):
    return np.array([sentiment(t)[1] for t in x]).reshape(-1, 1)

#       *******Length feature******
def get_text_length(x):
    return np.array([len(t) for t in x]).reshape(-1, 1)

def other(x):
    a = np.array([sentiment(t)[0] for t in x]).reshape(-1, 1)
    b = np.array([math.sqrt(len(t)) for t in x]).reshape(-1, 1)
    return [a*b for a,b in zip(a,b)]

def other2(x):
    a = np.array([sentiment(t)[1] for t in x]).reshape(-1, 1)
    b = np.array([math.sqrt(len(t)) for t in x]).reshape(-1, 1)
    return [a*b for a,b in zip(a,b)]

#		*******Feature pipeline******

pipeline = Pipeline([
    ('features_union', FeatureUnion([
                ('ngrams_feature', Pipeline([('ngrams_vect', TfidfVectorizer(binary= True, ngram_range=(1,3)))
            ])),
                ('words_feature', Pipeline([('words_vect', CountVectorizer(binary = True, vocabulary = vocab))
            ])),
                ('length',Pipeline([('count', FunctionTransformer(get_text_length, validate = False))
            ])),
                ('sent',Pipeline([('sentiment', FunctionTransformer(get_sentiment, validate = False))
            ])),
                ('sent2',Pipeline([('sentiment2', FunctionTransformer(get_sentiment2, validate = False))
            ])),
                ('other',Pipeline([('other', FunctionTransformer(other, validate = False))
            ])),
                ('other2',Pipeline([('other2', FunctionTransformer(other2, validate = False))
    ]))])),
    ('normalization', Normalizer(copy=False)),
    ('classifier', LinearSVC(penalty='l2'))])

#       *********Applying preprocessing*******
reviews = compile(reviews)


#		*********Grid Search*******
parameters_grid = {	'classifier__loss': ('hinge','squared_hinge'),
                    'classifier__C': (50,100,200),
                    'classifier__max_iter': (5000,8000),
					'features_union__words_feature__words_vect__max_df': (0.90, 0.95, 1),
                    'features_union__ngrams_feature__ngrams_vect__ngram_range': ((1,3),(2,3)),
					'features_union__transformer_weights': [dict(count=0),
															dict(words_vect=1, ngrams_vect=1),
															dict(other=0),
                                                            dict(sentiment=0),
                                                            dict(sentiment2=0),
															dict(other2=0)]
					}

		# *********Validation Pipeline*******

grid_search = GridSearchCV(pipeline, parameters_grid, cv=3, n_jobs=1, scoring='accuracy')
grid_search.fit(reviews,target)
cvres = grid_search.cv_results_
for accuracy, params in zip(cvres['mean_test_score'],cvres['params']):
	print('Mean accuracy: ', accuracy,'  using: ',params)