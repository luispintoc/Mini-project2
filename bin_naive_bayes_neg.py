import os, re, json
from random import shuffle
import numpy as np
from nltk.corpus import movie_reviews
import nltk as nl
import nltk.sentiment as nltk
import math as math

target = [] #len = 25000
reviews = [] #Shuffled training data, len = 25000
reviews_test = []
reviews_target = []

#test_path = 'data/test'
#for file in os.listdir(test_path):
#       with open(os.path.join(test_path,file),"r") as f:
#               reviews_test.append(f.read())
#print(reviews_test[1])
#print(len(reviews_test))
with open("train_data.json") as fp:
    train_data = json.load(fp)


i = 0
while i < 25000:
        target.append(train_data[i][0])
        reviews.append(train_data[i][1])
        i += 1

delete = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
replace_with_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocessing(reviews):
        reviews = [delete.sub("",line.lower()) for line in reviews]
        reviews = [replace_with_space.sub(" ",line) for line in reviews]
        cv = CountVectorizer(binary = True)



        return reviews  

#splits into train and validation sets
reviews_train = reviews[0:21250];
target_train = target[0:21250];
reviews_valid = reviews[21250:25000];
target_valid = target[21250:25000];
#print(reviews_train[1])
#print(target_train[1])

#reads positive words from Bing Liu's opinion lexicon
#text_file = open("positive-words.txt")
#pos_words = text_file.read().split()
#text_file.close()

#reads negative words
text_file = open("negative-words.txt")
neg_words = text_file.read().split()
text_file.close()

#Combines positive and negative words into one list
lexicon = neg_words

#Adds movie review list from nltk to list
#nl.download('movie_reviews')
#words_for_movies = list(set(list(movie_reviews.words())))
#words_for_movies = [str(r) for r in words_for_movies]
#lexicon = list(set().union(lexicon, words_for_movies))

#Splits training reviews into list of positive and negative reviews
pos_reviews = []
neg_reviews = []
counter = 0
for i in range(len(target_train)):
  if target_train[i]:
    counter = counter + 1
    pos_reviews.append(reviews_train[i])
  else:
    neg_reviews.append(reviews_train[i])
#Finds probability review is positive
Py1 = counter/float(len(target_train))
Py0 = 1 - Py1

#Finds P(xj = 1 | y =1) and P(xj = 0|y =1)
p_11 = [0] * len(lexicon)
p_01 = [0] * len(lexicon)
for i in pos_reviews:
  unigrams = nltk.util.extract_unigram_feats(i.split(),lexicon) #returns dictionary {contains(word): true or false} for each review
  for j in range(len(lexicon)):
    if (unigrams['contains('+lexicon[j]+')']):		
      p_11[j] = p_11[j] + 1
    else:
      p_01[j] = p_01[j] + 1

#compute probabilities      
p_11 = [(x + 1)/float((len(pos_reviews)+2)) for x in p_11]	#Laplace smoothed 
p_01 = [x / float(len(pos_reviews)) for x in p_01]

#Finds P(xj = 1 | y =0 and P(xj = 0|y =0)
p_00 = [0] * len(lexicon)
p_10 = [0] * len(lexicon)
for i in neg_reviews:
  unigrams = nltk.util.extract_unigram_feats(i.split(),lexicon)
  for j in range(len(lexicon)):
    if (unigrams['contains('+lexicon[j]+')']):
      p_10[j] = p_10[j] + 1
    else:
      p_00[j] = p_00[j] + 1

p_10 = [(x+1)/ float((len(neg_reviews)+2)) for x in p_10]	#compute probabilities
p_00= [x / float(len(neg_reviews)) for x in p_00]


#Makes decision boundary and classifies validation examples
predict = [0] * len(reviews_valid)
for i in range(len(reviews_valid)):
  unigrams = nltk.util.extract_unigram_feats(reviews_valid[i].split(),lexicon)
  boundary =  math.log10(Py1/Py0)
  for j in range(len(lexicon)):
    x_j = unigrams['contains('+lexicon[j]+')'];
    w_j0 = math.log10(p_01[j]/p_00[j])
    w_j1 = math.log10(p_11[j]/p_10[j])
    boundary = boundary + w_j0 + (w_j1 - w_j0)*x_j
  
  if boundary > 0:
    predict[i] = 1

#for i in range(len(predict)):
#  print i, ',', predict[i]

#Compares prediction with validation set
TP = 0
FP = 0
TN = 0
FN = 0
for i in range(len(reviews_valid)):
  if target_valid[i] == 1:
    if predict[i] == 1:
      TP = TP + 1
    else:
      FN = FN + 1
  else:
    if predict[i] == 1:
      FP = FP + 1
    else:
      TN = TN + 1
print("True positives: ",TP)
print("True negatives: ",TN)
print("False positives: ",FP)
print("False negatives: ",FN)

Error_rate = (FP + FN)/float(len(reviews_valid))
Accuracy = (TP + TN)/float(len(reviews_valid))
Precision = TP/float((TP + FP))
Recall = TP/float((TP + FN))
Specificity = TN/float((FP + TN))

print("Error Rate: ", Error_rate)
print("Accuracy: ", Accuracy)
print("Precision: ",Precision)
print("Recall: ",Recall)
print("Specificity: ",Specificity)

