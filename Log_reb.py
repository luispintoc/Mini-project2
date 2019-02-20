import os, re, json, math
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.corpus import stopwords, movie_reviews
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
# from pattern.en import sentiment
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
from sklearn.ensemble import VotingClassifier
import warnings
import nltk
warnings.filterwarnings("ignore",category = FutureWarning)

#       *****Initialization*****

target = [] #len = 25000
reviews = [] #Shuffled training data, len = 25000
positive_words = []
negative_words = []
bing_liu_list = []
stopwords = []

#       ********Read data*********

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
with open('stopwords.txt') as f:
    stopwords = f.read().splitlines()

bing_liu_list = list(zip(positive_words,negative_words))
words_for_movies = list(movie_reviews.words())
vocab = list(set(bing_liu_list + words_for_movies))

#       ********Preprocessing********

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


#       ********Features********

# def get_sentiment(x):
#     return np.array([sentiment(t)[0] for t in x]).reshape(-1, 1)

# def get_sentiment2(x):
#     return np.array([sentiment(t)[1] for t in x]).reshape(-1, 1)

#       *******Length feature******
def get_text_length(x):
    return np.array([len(t) for t in x]).reshape(-1, 1)

# def other(x):
#     a = np.array([sentiment(t)[0] for t in x]).reshape(-1, 1)
#     b = np.array([math.sqrt(len(t)) for t in x]).reshape(-1, 1)
#     return [a*b for a,b in zip(a,b)]

# def other2(x):
#     a = np.array([sentiment(t)[1] for t in x]).reshape(-1, 1)
#     b = np.array([math.sqrt(len(t)) for t in x]).reshape(-1, 1)
#     return [a*b for a,b in zip(a,b)]


#       *******Feature pipeline******
clf1 = LogisticRegression()
clf2 = LinearSVC()
eclf = VotingClassifier(estimators=[('lr',clf1),('svc',clf2)],voting='soft')


pipeline = Pipeline([
    ('features_union', FeatureUnion([
                ('ngrams_feature', Pipeline([('ngrams_vect', TfidfVectorizer(binary = True, ngram_range=(1,3)))
            ])),
                ('words_feature', Pipeline([('words_vect', CountVectorizer(vocabulary = vocab, binary = True, min_df = 2))
            ])),
                ('length',Pipeline([('count', FunctionTransformer(get_text_length, validate = False))
            # ])),
            #     ('sent',Pipeline([('sentiment', FunctionTransformer(get_sentiment, validate = False))
            # ])),
            #     ('sent2',Pipeline([('sentiment2', FunctionTransformer(get_sentiment2, validate = False))
            # ])),
            #     ('other',Pipeline([('other', FunctionTransformer(other, validate = False))
            # ])),
            #     ('other2',Pipeline([('other2', FunctionTransformer(other2, validate = False))
    ]))])),
        # ],
    #transformer_weights= {'words_feature': 1, 'ngrams_feature': 1,   }
    ('normalization', Normalizer(copy=False)),
    ('reduce_dim', None),
    #('classifier', LinearSVC(max_iter = 5000))])
    ('classifier', eclf)])
    #('classifier', LogisticRegression(solver= 'lbfgs', C= 100, penalty = 'l2', max_iter = 6000))])
    # ('classifier', RandomForestClassifier(n_estimators = 600))])
    #('classifier', LinearSVC(penalty='l2',C=100, max_iter = 5000))])

#       *********Applying preprocessing*******

reviews = compile(reviews)      #always apply this to get rid of punctuation and special characters

# reviews = get_stemmed_text(reviews,'Porter')
# reviews = get_stemmed_text(reviews,'Snow')
# reviews = get_lemmatized_text(reviews)


#       *********Grid Search*******
# call the labels in the pipeline above + __ + hyper-parameter for that label and in () indicate the different parameters to experiment
#print(pipeline.get_params().keys())
parameters_grid = { #'classifier__solver': ('lbfgs','newton-cg'),
                    'classifier__lr__C': (10,100,200),
                    'classifier__svc__C': (10,100,200),
                    
                    # 'reduce_dim__score_func':(f_classif(),chi2()),
                    # 'features_union__words_feature__words_vect__binary': (True,False),
                    # 'features_union__words_feature__words_vect__vocabulary': (vocab, positive_words, negative_words),
                    # 'features_union__ngrams_feature__ngrams_vect__binary': (True,False),
                    # 'features_union__transformer_weights': [dict(words_vect=0.5, ngrams_vect=10),
                    #                                       dict(words_vect=2, ngrams_vect=5),
                    #                                       dict(words_vect=5, ngrams_vect=2),
                    #                                       dict(words_vect=10, ngrams_vect=0.5)]
                    'reduce_dim':[SelectKBest(f_classif)],
                    # 'reduce_dim__score_func':(f_classif(),chi2()),
                    'reduce_dim__k':(10000,11000,20000,15000,17500)}

        # *********Validation Pipeline*******

grid_search = GridSearchCV(pipeline, parameters_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid_search.fit(reviews,target)
cvres = grid_search.cv_results_
for accuracy, params in zip(cvres['mean_test_score'],cvres['params']):
    print('Mean accuracy: ', accuracy,'  using: ',params)


# x_train,x_val,y_train,y_val = train_test_split(compile(reviews), target, train_size = 0.80, random_state = 45)
# x_train = compile(x_train)
# x_val = compile(x_val)        #always apply this to get rid of punctuation and special characters

# pipeline.fit(x_train,y_train)
# print("Accuracy on train set: %s " %(accuracy_score(y_train,pipeline.predict(x_train))))
# print("Accuracy on val set: %s " %(accuracy_score(y_val,pipeline.predict(x_val))))

