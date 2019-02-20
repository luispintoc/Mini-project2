import os, re, json, math
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.corpus import stopwords, movie_reviews
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
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

bing_liu_list = list(zip(positive_words,negative_words))
# words_for_movies = list(movie_reviews.words())
# vocab = list(set(bing_liu_list + words_for_movies))

#		********Preprocessing********

delete = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
replace_with_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def compile(reviews):
	reviews = [delete.sub("",line.lower()) for line in reviews]
	reviews = [replace_with_space.sub(" ",line) for line in reviews]
	return reviews

#		*******Feature pipeline******

pipeline = Pipeline([
    ('features_union', FeatureUnion([
                ('ngrams_feature', Pipeline([('ngrams_vect', TfidfVectorizer(binary = True, ngram_range=(1,3)))
            # ])),
            #     ('words_feature', Pipeline([('words_vect', CountVectorizer(min_df = 2, binary = True, vocabulary = bing_liu_list))
    ]))])),
    ('normalization', Normalizer(copy=False)),
    ('reduce_dim', None),
    ('classifier', RandomForestClassifier())])

#       *********Applying preprocessing*******

reviews = compile(reviews)		#always apply this to get rid of punctuation and special characters

#		*********Grid Search*******

#print(pipeline.get_params().keys())
parameters_grid = {	'classifier__n_estimators': (10,100,200,500),
                    'classifier__min_samples_split':(2,3,4),
                    'classifier__max_features': (0.7,0.4),
                    'reduce_dim':[SelectKBest(f_classif)],
                    'reduce_dim__k':(60000,65000,75000,80000,100000,110000)}

		# *********Validation Pipeline*******

print('Running gridseach on Log for only vect features')


grid_search = GridSearchCV(pipeline, parameters_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid_search.fit(reviews,target)
cvres = grid_search.cv_results_
for accuracy, params in zip(cvres['mean_test_score'],cvres['params']):
	print('Mean accuracy: ', accuracy,'  using: ',params)
