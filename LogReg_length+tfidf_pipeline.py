import os, re, json, math
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore",category = FutureWarning)


#		*****Initialization*****

bing_liu_list = []
target = [] #len = 25000
reviews = [] #Shuffled training data, len = 25000
positive_words = []
negative_words = []
length_list = []


#		********Read data*********

with open("train_data2.json") as fp:
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

#		********Preprocessing********

delete = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
replace_with_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def compile(reviews):
	reviews = [delete.sub("",line.lower()) for line in reviews]
	reviews = [replace_with_space.sub(" ",line) for line in reviews]
	return reviews

def vectorization(train,test, tfidf):	#It will vectorize the train set and it will transform both train and test set
	if tfidf == "tfidf_on":
		cv = TfidfVectorizer(binary = False, min_df = 2, ngram_range=(1,2))
		cv.fit(train)
		train = cv.transform(train)
		test = cv.transform(test)
	else:
		cv = CountVectorizer(binary = True, min_df = 30, ngram_range=(1,2))
		cv.fit(train)
		train = cv.transform(train)
		test = cv.transform(test)
	return train, test

def get_stemmed_text(corpus,name): #PorterStemmer - SnowballStemmer("english")
	if name == 'Porter':
		from nltk.stem.porter import PorterStemmer
		stemmer = PorterStemmer()
	else:
		from nltk.stem.snowball import SnowballStemmer
		stemmer = SnowballStemmer("english")
	return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]


def normalization(train,test):
	norm = Normalizer().fit(train)
	train = norm.transform(train)
	test = norm.transform(test)
	return train, test

#		*******Length feature******
def get_text_length(x):
    return np.array([math.sqrt(len(t)) for t in x]).reshape(-1, 1)


#       *********Features Pipeline*******

pipeline = Pipeline([
    ('features_union', FeatureUnion([
            ('ngrams_feature', Pipeline([('ngrams_vect', TfidfVectorizer(binary=False, ngram_range=(1,2))),
            ])),
    ('length',Pipeline([
    		('count', FunctionTransformer(get_text_length, validate = False)),
    ]))])),
        # ],
    #transformer_weights= {'words_feature': 1, 'ngrams_feature': 1,   }
    #('normalization', Normalizer(copy=False)),
    ('classifier', LogisticRegression(C=100, penalty = 'l2'))])


#       *********Applying preprocessing*******

reviews = compile(reviews)		#always apply this to get rid of punctuation and special characters

#reviews = get_stemmed_text(reviews,'Porter')
#reviews = get_stemmed_text(reviews,'Snow')
#reviews = get_lemmatized_text(reviews)


#		*********Grid Search*******
#call the labels in the pipeline above + __ + hyper-parameter for that label and in () indicate the different parameters to experiment
#print(pipeline.get_params().keys())
parameters_grid = {	#'classifier__C': (50,100,150),
					'features_union__words_feature__words_vect__max_features': (150,350,500),
					'features_union__transformer_weights': [dict(words_vect=0.5, ngrams_vect=10),
															dict(words_vect=2, ngrams_vect=5),
															dict(words_vect=5, ngrams_vect=2),
															dict(words_vect=10, ngrams_vect=0.5)]
					}

#		*********Validation Pipeline*******

grid_search = GridSearchCV(pipeline, parameters_grid, cv=4, n_jobs=-2, scoring='accuracy')
grid_search.fit(reviews,target)
cvres = grid_search.cv_results_
for accuracy, params in zip(cvres['mean_test_score'],cvres['params']):
	print('Mean accuracy: ', accuracy,'  using: ',params)
