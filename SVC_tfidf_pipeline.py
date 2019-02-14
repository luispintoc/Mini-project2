import os, re, json
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
import warnings

warnings.filterwarnings("ignore",category = FutureWarning)
#warnings.filterwarnings("ignore")

#		*****Initialization*****

filter_words = []
target = [] #len = 25000
reviews = [] #Shuffled training data, len = 25000


#		********Read data*********

with open("train_data2.json") as fp:
    train_data = json.load(fp)
i = 0
while i < 25000:
	target.append(train_data[i][0])
	reviews.append(train_data[i][1])
	i += 1


#		********Preprocessing********

delete = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
replace_with_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def compile(reviews):
	reviews = [delete.sub("",line.lower()) for line in reviews]
	reviews = [replace_with_space.sub(" ",line) for line in reviews]
	return reviews


def get_stemmed_text(corpus,name): #PorterStemmer - SnowballStemmer("english")
	if name == 'Porter':
		from nltk.stem.porter import PorterStemmer
		stemmer = PorterStemmer()
	else:
		from nltk.stem.snowball import SnowballStemmer
		stemmer = SnowballStemmer("english")
	return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

def get_lemmatized_text(corpus):
    import nltk
    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus] 

def normalization(train):
	norm = Normalizer().fit(train)
	train = norm.transform(train)
	return train


#		*******Feature pipelines******

vect_pipeline = Pipeline([('vect', CountVectorizer(ngram_range = (1,2),binary=False)),
							('tfidf', TfidfTransformer()),
							#('normalization', (copy = False)), #Normalizer
							('classifier', SVC(max_iter=2000))])

#       *********Applying preprocessing*******

reviews = compile(reviews)
#reviews = normalization(reviews)
#x_train,x_val,y_train,y_val = train_test_split(compile(reviews), target, train_size = 0.75, random_state = 42)
# x_train = get_stemmed_text(x_train,'Porter')
# x_val = get_stemmed_text(x_val,'Porter')
#x_train = get_stemmed_text(x_train,'Snow')
#x_val = get_stemmed_text(x_val,'Snow')
# x_train = get_lemmatized_text(x_train)
# x_val = get_lemmatized_text(x_val)
#[x_train,x_val] = tf_idf_vectorization(x_train, x_val)


#		*********Grid Search*******
parameters_grid = {#'vect__binary': (True,False),
					#'classifier__decision_function_shape':('ovo','ovr'),
					'classifier__C':(1,100)
					}


#		*********Validation Pipeline*******

grid_search = GridSearchCV(vect_pipeline, parameters_grid, cv=4, n_jobs=-2, scoring='accuracy')
grid_search.fit(reviews,target)
cvres = grid_search.cv_results_
for accuracy, params in zip(cvres['mean_test_score'],cvres['params']):
	print('Mean accuracy: ', accuracy,'  using: ',params)