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

with open("train_data.json") as fp:
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


#		*******Length feature******
def get_text_length(x):
    return np.array([math.sqrt(len(t)) for t in x]).reshape(-1, 1)

# length_feature = FunctionTransformer(length)
# print(length_feature.fit_transform(compile(reviews)[0:10]).shape)
#print(compile(reviews)[0])
#print(length(compile(reviews)[0:10]))

def normalization(train,test):
	norm = Normalizer().fit(train)
	train = norm.transform(train)
	test = norm.transform(test)
	return train, test



#		********Heald-out validation********

x_train,x_val,y_train,y_val = train_test_split(compile(reviews), target, train_size = 0.80, random_state = 42)


#       *********Applying preprocessing*******
# x_train = get_stemmed_text(x_train,'Snow')
# x_val = get_stemmed_text(x_val,'Snow')


#       *********Features Pipeline*******

## Define pipeline and call classifier for multiple max_depths
for i in range(15):
  for j in range(10):
    pipeline = Pipeline([
       ('features_union', FeatureUnion([
                ('ngrams_feature', Pipeline([('ngrams_vect', TfidfVectorizer(binary=False, ngram_range=(1,2))),
                ])),
        ('length',Pipeline([
      	  	('count', FunctionTransformer(get_text_length, validate = False)),
        ]))])),
        # ],
     
	#transformer_weights={'words_feature': 1, 'ngrams_feature': 1,   }
    #('normalization', Normalizer(copy=False)),
        ('classifier', DecisionTreeClassifier(max_depth=i+5, min_samples_split=j+2))])


#		*********Classifiers*******

    print('Classifier: Decision Trees with max_depth',i+5,' and min_samples_split ',j+2)
    pipeline.fit(x_train, y_train)
    print("Accuracy on train set: %s " %(accuracy_score(y_train,pipeline.predict(x_train))))
    print("Accuracy on val set: %s " %(accuracy_score(y_val,pipeline.predict(x_val))))


#       *********Applying snow preprocessing*******
x_train = get_stemmed_text(x_train,'Snow')
x_val = get_stemmed_text(x_val,'Snow')

for i in range(15):
  for j in range(10):
    pipeline = Pipeline([
       ('features_union', FeatureUnion([
                ('ngrams_feature', Pipeline([('ngrams_vect', TfidfVectorizer(binary=False, ngram_range=(1,2))),
                ])),
        ('length',Pipeline([
                ('count', FunctionTransformer(get_text_length, validate = False)),
        ]))])),
        # ],

        #transformer_weights={'words_feature': 1, 'ngrams_feature': 1,   }
    #('normalization', Normalizer(copy=False)),
        ('classifier', DecisionTreeClassifier(max_depth=i+5, min_samples_split=j+2))])


#               *********Classifiers*******

    print('Classifier: Decision Trees, snow preprocessing, with max_depth',i+5,' and min_samples_split ',j+2)
    pipeline.fit(x_train, y_train)
    print("Accuracy on train set: %s " %(accuracy_score(y_train,pipeline.predict(x_train))))
    print("Accuracy on val set: %s " %(accuracy_score(y_val,pipeline.predict(x_val))))

#       *********Applying preprocessing*******
x_train = get_stemmed_text(x_train,'Porter')
x_val = get_stemmed_text(x_val,'Porter')

for i in range(15):
  for j in range(10):
    pipeline = Pipeline([
       ('features_union', FeatureUnion([
                ('ngrams_feature', Pipeline([('ngrams_vect', TfidfVectorizer(binary=False, ngram_range=(1,2))),
                ])),
        ('length',Pipeline([
                ('count', FunctionTransformer(get_text_length, validate = False)),
        ]))])),
        # ],

        #transformer_weights={'words_feature': 1, 'ngrams_feature': 1,   }
    #('normalization', Normalizer(copy=False)),
        ('classifier', DecisionTreeClassifier(max_depth=i+5, min_samples_split=j+2))])


#               *********Classifiers*******

    print('Classifier: Decision Trees with Porter Preprocessing max_depth',i+5,' and min_samples_split ',j+2)
    pipeline.fit(x_train, y_train)
    print("Accuracy on train set: %s " %(accuracy_score(y_train,pipeline.predict(x_train))))
    print("Accuracy on val set: %s " %(accuracy_score(y_val,pipeline.predict(x_val))))
