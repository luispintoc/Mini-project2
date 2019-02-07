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
import warnings
warnings.filterwarnings("ignore",category = FutureWarning)
filter_words = []

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

def vectorization(train,test, words):			#It will vectorize the train set and it will transform both train and test set
	cv = CountVectorizer(binary = True, stop_words = words)
	cv.fit(train)
	train = cv.transform(train)
	test = cv.transform(test)
	return train, test

# stop_words = stopwords.words(words)
# def filter_stopwords(corpus):
#     filter_stopwords = []
#     for review in corpus:
#         filter_stopwords.append(
#             ' '.join([word for word in review.split() 
#                       if word not in stop_words])
#         )
#     return filter_stopwords


#		*********Validation Pipeline******
# def validation(classifier,x,y,k_fold):
# 	accuracy = cross_val_score(classifier,x,y,cv = k_fold)
# 	print('Accuracies for a ',k_fold,'fold cross validation:')
# 	return accuracy


#		*********Classifiers*******

x_train,x_val,y_train,y_val = train_test_split(compile(reviews), target, train_size = 0.75, random_state = 42)
[x_train,x_val] = vectorization(x_train, x_val, filter_words)

print('Classifier: LogisticRegression')
# for c in [0.01, 0.05, 0.25, 0.5, 1]:
# 	lr = LogisticRegression(C=c)
# 	lr.fit(x_train,y_train)
# 	print("Accuracy for C=%s: %s" %(c,accuracy_score(y_val,lr.predict(x_val))))

lr = LogisticRegression(C = 0.05)
lr.fit(x_train,y_train)
print("Accuracy using C = 0.05: %s " %(accuracy_score(y_val,lr.predict(x_val))))


print('Classifier: Support Vector Machines')
# for c in [0.01, 0.05, 0.25, 0.5, 1]:
# 	svm = LinearSVC(C = c)
# 	svm.fit(x_train,y_train)
# 	print("Accuracy for C=%s: %s" %(c,accuracy_score(y_val,svm.predict(x_val))))

svm = LinearSVC(C = 0.01)
svm.fit(x_train,y_train)
print("Accuracy using C = 0.01: %s " %(accuracy_score(y_val,svm.predict(x_val))))

print('Classifier: Random Forest')
forest_reg = RandomForestClassifier()
forest_reg.fit(x_train,y_train)
print("Accuracy: ", accuracy_score(y_val,forest_reg.predict(x_val)))

print('Classifier: Decision Tree')
tree_reg = DecisionTreeClassifier()
tree_reg.fit(x_train,y_train)
print("Accuracy: ", accuracy_score(y_val,tree_reg.predict(x_val)))



#		*******Extra*****

#returns the 10 most discriminative words for positive and negative reviews
#which can be used to increase predictivity
# feature_to_coef = {
#     word: coef for word, coef in zip(
#         CountVectorizer(binary=True).get_feature_names(), lr.coef_[0]
#     )
# }
# for best_positive in sorted(
#     feature_to_coef.items(), 
#     key=lambda x: x[1], 
#     reverse=True)[:10]:
#     print (best_positive)

# for best_negative in sorted(
#     feature_to_coef.items(), 
#     key=lambda x: x[1])[:10]:
#     print (best_negative)


