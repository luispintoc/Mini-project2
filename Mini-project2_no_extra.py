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


#		******Read data******

for file in os.listdir(train_pos_path):
	with open(os.path.join(train_pos_path,file),"r",encoding="utf8") as f:
		reviews_train.append(f.read())

for file in os.listdir(train_neg_path):
	with open(os.path.join(train_neg_path,file),"r",encoding="utf8") as f:
		reviews_train.append(f.read())

target = [1 if i<12500 else 0 for i in range(25000)]
reviews = reviews_train 
shuffle(reviews)


#		********Preprocessing********

delete = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
replace_with_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def compile(reviews):
	reviews = [delete.sub("",line.lower()) for line in reviews]
	reviews = [replace_with_space.sub(" ",line) for line in reviews]
	return reviews

def vectorization(train,test, words):	#It will vectorize the train set and it will transform both train and test set
	cv = CountVectorizer(binary = True, stop_words = words, min_df = 0.01)
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
x_train = get_stemmed_text(x_train,'Snow')
x_val = get_stemmed_text(x_val,'Snow')
[x_train,x_val] = vectorization(x_train, x_val, filter_words)


#		*********Classifiers*******

print('Classifier: LogisticRegression')
lr = LogisticRegression(C = 0.05)
lr.fit(x_train,y_train)
print("Accuracy on train set: %s " %(accuracy_score(y_train,lr.predict(x_train))))
print("Accuracy on val set: %s " %(accuracy_score(y_val,lr.predict(x_val))))


print('Classifier: Support Vector Machines')
svm = LinearSVC(C = 0.01)
svm.fit(x_train,y_train)
print("Accuracy on train set: %s " %(accuracy_score(y_train,svm.predict(x_train))))
print("Accuracy on val set: %s " %(accuracy_score(y_val,svm.predict(x_val))))
