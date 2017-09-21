import nltk
from nltk.stem.porter import *
from nltk.classify import MaxentClassifier
import pickle
import os,sys
from io import open
import string
import time
import re

# reload(sys)
# sys.setdefaultencoding('utf8')

print("start program ", time.strftime('%X %x %Z'))

# from itertools import izip
from itertools import groupby



bookname_list = ['not', 'book']
labeled_features = []

training_file = open("../chat_history/training_questions.txt", "r")
training_label_file = open("../chat_history/training_slots.txt", "r")

def splitWithIndices(s, c=' '):
    p = 0
    for k, g in groupby(s, lambda x:x==c):
        q = p + sum(1 for i in g)
        if not k:
            yield p, q # or p, q-1 if you are really sure you want that
        p = q


def MEMM_features(two_word, one_word, word, word_one, word_two, previous_tag):

    features = {}

    features['previous2_word'] = two_word
    features['previous_word'] = one_word
    features['current_word'] = word
    features['last_word'] = word_one
    features['last2_word'] = word_two
    features['capitalization'] = word[0].isupper()
    features['previous_tag'] = previous_tag
#    features['query_word2'] = query_word2
#    features['query_word'] = query_word

    if 'q' == features['previous_tag']:
        print (features)
    return features



for x, y in zip(training_file, training_label_file):
    x = x.strip()
    # x = x.translate(None, string.punctuation)
    x = re.sub(r'[^\w\s]', '', x)
    y = y.strip()
    # y = y.translate(None, string.punctuation)
    y = re.sub(r'[^\w\s]', '', y)

    y_len = len(y)
    start_index = x.find(y)

    end_index = start_index + y_len
    token_indices = list(splitWithIndices(x))
    book_list = []
    query_list = ['interested','interest','learn','research','know','about','related','show','by','written','find']
    for i in range(len(token_indices)):
        if token_indices[i][0] >= start_index and token_indices[i][1] <= end_index:
            book_list.append(i)
    sentenceList = x.split()
    start_word = "startttt"
    end_word = "endddd"
#print(book_list)
#   print(sentenceList)
    for ii in range(2):
        sentenceList.insert(0,start_word)
        sentenceList.append(end_word)

    for iii in range(2, len(sentenceList)-2):
        if iii-2 in book_list:
            if iii-3 in book_list:
                item = sentenceList[iii-2], sentenceList[iii-1], sentenceList[iii], sentenceList[iii+1], sentenceList[iii+2], "book", "book"
                labeled_features.append(item)
            elif sentenceList[iii-3-1] in query_list:
                item = sentenceList[iii-2], sentenceList[iii-1], sentenceList[iii], sentenceList[iii+1], sentenceList[iii+2], "q", "book"
                labeled_features.append(item)
            else:
                item = sentenceList[iii-2], sentenceList[iii-1], sentenceList[iii], sentenceList[iii+1], sentenceList[iii+2], "not", "book"
                labeled_features.append(item)
        elif sentenceList[iii-2-1] in query_list:
            if sentenceList[iii-3] in query_list:
                item = sentenceList[iii-2], sentenceList[iii-1], sentenceList[iii], sentenceList[iii+1], sentenceList[iii+2], "q", "q"
                labeled_features.append(item)
            else:
                item = sentenceList[iii-2], sentenceList[iii-1], sentenceList[iii], sentenceList[iii+1], sentenceList[iii+2], "not", "q"
                labeled_features.append(item)

        else:
            if iii-3 in book_list:
                item = sentenceList[iii-2], sentenceList[iii-1], sentenceList[iii], sentenceList[iii+1], sentenceList[iii+2], "book", "not"
                labeled_features.append(item)
            elif sentenceList[iii-3-1] in query_list:
                item = sentenceList[iii-2], sentenceList[iii-1], sentenceList[iii], sentenceList[iii+1], sentenceList[iii+2], "q", "not"
                labeled_features.append(item)
            else:
                item = sentenceList[iii-2], sentenceList[iii-1], sentenceList[iii], sentenceList[iii+1], sentenceList[iii+2], "not", "not"
                labeled_features.append(item)
#print(labeled_features)
#print(book_list)
#print(sentenceList)
train_set = [(MEMM_features(two_word, one_word, word, word_one, word_two, previous_tag), real_tag )for (two_word, one_word, word, word_one, word_two, previous_tag, real_tag) in labeled_features]
print("start training", time.strftime('%X %x %Z'))
f = open("my_classifier_train.pickle", "wb")
maxent_classifier = MaxentClassifier.train(train_set, max_iter=30)
pickle.dump(maxent_classifier , f)
f.close()
print("finish training", time.strftime('%X %x %Z'))
