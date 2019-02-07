import os, re, json
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

target = [] #len = 25000
reviews = [] #Shuffled training data, len = 25000

with open("train_data.json") as fp:
    train_data = json.load(fp)
#print(target[:20])

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

def vectorization(train,test):			#It will vectorize the train set and it will transform both train and test set
	cv = CountVectorizer(binary = True)
	cv.fit(train)
	train = cv.transform(train)
	test = cv.transform(test)



	return reviews

