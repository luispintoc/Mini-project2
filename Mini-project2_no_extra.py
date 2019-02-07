import os, re, json
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore",category = FutureWarning)

target = [] #len = 25000
reviews = [] #Shuffled training data, len = 25000

with open("train_data.json") as fp:
    train_data = json.load(fp)


i = 0
while i < 25000:
	target.append(train_data[i][0])
	reviews.append(train_data[i][1])
	i += 1
#print(target[:20])


#		********Preprocessing********

delete = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
replace_with_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def compile(reviews):
	reviews = [delete.sub("",line.lower()) for line in reviews]
	reviews = [replace_with_space.sub(" ",line) for line in reviews]
	return reviews

def vectorization(train,test):			#It will vectorize the train set and it will transform both train and test set
	cv = CountVectorizer(binary = True)
	cv.fit(train)
	train = cv.transform(train)
	test = cv.transform(test)
	return train, test

#		*********Validation Pipeline******
# def validation(classifier,x,y,k_fold):
# 	accuracy = cross_val_score(classifier,x,y,cv = k_fold)
# 	print('Accuracies for a ',k_fold,'fold cross validation:')
# 	return accuracy


#		*********Classifiers*******

x_train,x_val,y_train,y_val = train_test_split(compile(reviews), target, train_size = 0.75, random_state = 42)
[x_train,x_val] = vectorization(x_train,x_val)

print('Classifier: LogisticRegression')
for c in [0.01, 0.05, 0.25, 0.5, 1]:
	lr = LogisticRegression(C=c)
	lr.fit(x_train,y_train)
	print("Accuracy for C = %s: %s" %(c,accuracy_score(y_val,lr.predict(x_val))))
