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
from sklearn.preprocessing import Normalizer, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings("ignore",category = FutureWarning)

#		*****Initialization*****

target = [] #len = 25000
reviews = [] #Shuffled training data, len = 25000
positive_words = []
negative_words = []
bing_liu_list = []

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


#		*******Feature pipeline******

pipeline = Pipeline([
    ('features_union', FeatureUnion(
        transformer_list = [
            ('words_feature', Pipeline([('words_vect', CountVectorizer(stop_words='english')),
            ])),
            ('ngrams_feature', Pipeline([('ngrams_vect', TfidfVectorizer(ngram_range=(1,2))),
            ]))
        ]#,
        #transformer_weights= {'words_feature': 0.5, 'ngrams_feature': 1,   }
    )),
    ('normalization',Normalizer(copy=False)),
    ('classifier', LogisticRegression(penalty = 'l2', C=100)),
])

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








'''





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