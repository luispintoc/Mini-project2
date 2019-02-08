import os, re, json
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
from sklearn.preprocessing import Normalizer
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


#		*******Feature pipelines******

vect_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
							('tfidf', TfidfTransformer()),
							('classifier', LogisticRegression())])

#       *********Applying preprocessing*******

reviews = compile(reviews)
#x_train,x_val,y_train,y_val = train_test_split(compile(reviews), target, train_size = 0.75, random_state = 42)
# x_train = get_stemmed_text(x_train,'Porter')
# x_val = get_stemmed_text(x_val,'Porter')
x_train = get_stemmed_text(x_train,'Snow')
x_val = get_stemmed_text(x_val,'Snow')
# x_train = get_lemmatized_text(x_train)
# x_val = get_lemmatized_text(x_val)
#[x_train,x_val] = tf_idf_vectorization(x_train, x_val)


#		*********Grid Search*******
parameters_grid = {'vect__binary': (True,False),
					'vect__min_df':(30,50,100,120),
					'tfidf__use_idf': (True, False),
					'classifier__penalty':('l1','l2'),
					'classifier__C':(0.05,0.15)}
# parameters_grid = {'vect__binary': (True,False),
# 					}

#		*********Validation Pipeline*******

grid_search = GridSearchCV(vect_pipeline, parameters_grid, cv=4, n_jobs=-2, scoring='accuracy')
grid_search.fit(reviews,target)
cvres = grid_search.cv_results_
for accuracy, params in zip(cvres['mean_test_score'],cvres['params']):
	print('Mean accuracy: ', accuracy,'  using: ',params)















'''
#		***********Code for the other classifiers**********


print('Classifier: Support Vector Machines')
# for c in [0.01, 0.05, 0.25, 0.5, 1]:
# 	svm = LinearSVC(C = c)
# 	svm.fit(x_train,y_train)
# 	print("Accuracy for C=%s: %s" %(c,accuracy_score(y_val,svm.predict(x_val))))

svm = LinearSVC(C = 0.01)

#svm = Pipeline((("scaler", Normalizer()),("linear_svc", LinearSVC(C=0.01, loss="hinge")),))
svm.fit(x_train,y_train)
print("Accuracy on train set using C = 0.01: %s " %(accuracy_score(y_train,svm.predict(x_train))))
print("Accuracy on val set using C = 0.01: %s " %(accuracy_score(y_val,svm.predict(x_val))))

# print('Classifier: Random Forest')
# forest_reg = RandomForestClassifier()
# forest_reg.fit(x_train,y_train)
# print("Accuracy on train set using C = 0.01: %s " %(accuracy_score(y_train,forest_reg.predict(x_train))))
# print("Accuracy on val set: ", accuracy_score(y_val,forest_reg.predict(x_val)))

# print('Classifier: Decision Tree')
# tree_reg = DecisionTreeClassifier()
# tree_reg.fit(x_train,y_train)
# print("Accuracy on train set: %s " %(accuracy_score(y_train,tree_reg.predict(x_train))))
# print("Accuracy on val set: ", accuracy_score(y_val,tree_reg.predict(x_val)))



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
'''
