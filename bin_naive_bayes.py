#load json file
from google.colab import files
files.upload()
# choose the file on your computer to upload it then

# Note that this does not work on Firefox for some reason, according to stack overflow

#load positive word list
from google.colab import files
files.upload()

import os, re, json
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

target = [] #len = 25000
reviews = [] #Shuffled training data, len = 25000

with open("train_data.json") as fp:
    train_data = json.load(fp)


i = 0
while i < 25000:
        target.append(train_data[i][0])
        reviews.append(train_data[i][1])
        i += 1

print(len(target))
print(len(reviews))


delete = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
replace_with_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocessing(reviews):
        reviews = [delete.sub("",line.lower()) for line in reviews]
        reviews = [replace_with_space.sub(" ",line) for line in reviews]
        cv = CountVectorizer(binary = True)



        return reviews  

#lists positive words from Bing Liu's opinion lexicon
text_file = open("positive-words.txt")
pos_words = text_file.read().split()
print(pos_words)
text_file.close()

import nltk.sentiment as nltk
#ocument = 'a+ abound abounds abundance acclamation'.split()
#nigrams = nltk.util.extract_unigram_feats(document, pos_words)
#print(unigrams[1])

#Finds probability of y
pos_reviews = []
neg_reviews = []
counter = 0
for i in target:
  if i:
    counter = counter + 1
    pos_reviews.append(reviews[i])
  else:
    neg_reviews.append(reviews[i])
Py = counter/len(target)

#Finds P(xj = 1 | y =1) and P(xj = 0|y =1)
p_11 = [0] * len(pos_words)
p_01 = [0] * len(pos_words)
reviews_test = []
print(len(reviews))
for i in range(25):
  reviews_test.append(pos_reviews[i])
for i in reviews_test:
  unigrams = nltk.util.extract_unigram_feats(i,pos_words)
  for j in range(len(pos_words)):
    if (unigrams['contains('+pos_words[j]+')']):
      p_11[j] = p_pos[j] + 1
    else:
      p_01[j] = p_pos[j] + 1
      
p_11 = [x / len(reviews_test) for x in p_11]
p_01 = [x / len(reviews_test) for x in p_01]

#Finds P(xj = 1 | y =0 and P(xj = 0|y =0
p_00 = [0] * len(pos_words)
p_10 = [0] * len(pos_words)
reviews_test = []
for i in range(25):
  reviews_test.append(neg_reviews[i])
for i in reviews_test:
  unigrams = nltk.util.extract_unigram_feats(i,pos_words)
  for j in range(len(pos_words)):
    if (unigrams['contains('+pos_words[j]+')']):
      p_10[j] = p_pos[j] + 1
    else:
      p_00[j] = p_pos[j] + 1

p_10 = [x / len(reviews_test) for x in p_10]
p_00= [x / len(reviews_test) for x in p_00]
