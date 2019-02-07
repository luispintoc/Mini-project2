import os, json
from random import shuffle
import numpy as np

reviews_train = []  #first 12500 are positive reviews, the rest are negative
reviews_test = []
train_pos_path = 'train//pos'
train_neg_path = 'train//neg'
test_path = 'test'
train_shuffled = open('train_shuffled.txt','w')


for file in os.listdir(train_pos_path):
	with open(os.path.join(train_pos_path,file),"r",encoding="utf8") as f:
		reviews_train.append(f.read())

for file in os.listdir(train_neg_path):
	with open(os.path.join(train_neg_path,file),"r",encoding="utf8") as f:
		reviews_train.append(f.read())
#print(len(reviews_train))

# for file in os.listdir(test_path):
# 	with open(os.path.join(test_path,file),"r",encoding="utf8") as f:
# 		reviews_test.append(f.read())
#print(len(reviews_test))

target_train = [1 if i<12500 else 0 for i in range(25000)]

train_set = list(zip(target_train,reviews_train))
shuffle(train_set)

with open('train_data.json','w') as outfile:
	json.dump(train_set,outfile)
