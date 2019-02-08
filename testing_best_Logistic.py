import os, re, json
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore",category = FutureWarning)


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

def vectorization(train,test, words):	#It will vectorize the train set and it will transform both train and test set
	cv = TfidfVectorizer(binary = True, min_df = 30, ngram_range=(1,2))
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


#		********Heald-out validation********

x_train,x_val,y_train,y_val = train_test_split(compile(reviews), target, train_size = 0.75, random_state = 42)


#       *********Applying preprocessing*******
# x_train = get_stemmed_text(x_train,'Snow')
# x_val = get_stemmed_text(x_val,'Snow')
[x_train,x_val] = vectorization(x_train, x_val, filter_words)


#		*********Classifiers*******

print('Classifier: LogisticRegression')
lr = LogisticRegression(C = 0.15, penalty = 'l2')
lr.fit(x_train,y_train)
print("Accuracy on train set: %s " %(accuracy_score(y_train,lr.predict(x_train))))
print("Accuracy on val set: %s " %(accuracy_score(y_val,lr.predict(x_val))))
