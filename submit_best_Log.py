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
import csv
import natsort
import warnings
warnings.filterwarnings("ignore",category = FutureWarning)


#		*****Initialization*****

filter_words = []
target = [] #len = 25000
reviews_train = []  #first 12500 are positive reviews, the rest are negative
reviews_test = []
train_pos_path = 'train//pos'
train_neg_path = 'train//neg'
test_path = 'test'
y_test = []


#		******Read data******

for file in os.listdir(train_pos_path):
	with open(os.path.join(train_pos_path,file),"r",encoding="utf8") as f:
		reviews_train.append(f.read())

for file in os.listdir(train_neg_path):
	with open(os.path.join(train_neg_path,file),"r",encoding="utf8") as f:
		reviews_train.append(f.read())

for file in natsort.natsorted(os.listdir(test_path),reverse=False):
	with open(os.path.join(test_path,file),"r",encoding="utf8") as f:
		reviews_test.append(f.read())

y_train = [1 if i<12500 else 0 for i in range(25000)]


#		********Preprocessing********

delete = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
replace_with_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def compile(reviews):
	reviews = [delete.sub("",line.lower()) for line in reviews]
	reviews = [replace_with_space.sub(" ",line) for line in reviews]
	return reviews

def vectorization(train,test):	#It will vectorize the train set and it will transform both train and test set
	cv = TfidfVectorizer(binary = False, min_df = 120, ngram_range=(1,2))
	cv.fit(train)
	train = cv.transform(train)
	test = cv.transform(test)
	return train, test

#		********Heald-out validation********

x_train = compile(reviews_train)
x_test = compile(reviews_test)


#       *********Applying preprocessing*******
[x_train,x_test] = vectorization(x_train, x_test)


#		*********Classifiers*******
print('Classifier: LogisticRegression')
lr = LogisticRegression(C = 0.15, penalty = 'l2')
lr.fit(x_train,y_train)
y_test = lr.predict(x_test)
print(y_test)

id = list(range(25000))
ss = list(zip(id,y_test))

with open('submission_log.csv', 'w', newline = '') as f:
     writer = csv.writer(f, delimiter=',')
     writer.writerows(ss)
