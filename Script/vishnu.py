"""
TODO:

creates dataset => find all slots for hard code slots

add punutation
add context free grammer 
add pos tag 
add trigram or bigram
captilaize of certain word

words not seen before
2 layers viterbi

other slot detection:
books

research
a word that contains multitples slots ????!!!!! => universal slot detection for library

transition probability (HMM, CRF) is suitable for limited amount + distinctive state with strong transition probability
hardest part is to divide real world problem into such states

"""

import nltk
from nltk.stem.porter import *
from nltk.classify import MaxentClassifier
import pickle
import os,sys
from io import open
import string
import time

reload(sys)
sys.setdefaultencoding('utf8')

print("start program ", time.strftime('%X %x %Z'))

from itertools import izip
from itertools import groupby

# bookname_list = ['not', 'LIBRARY', 'GRADUATE', 'FACILITY', 'DATE', 'MEMBERSHIP', 'PURPOSE', 'SUBJECT', 'LOCATION', 'UNIVERSITY', 'STUDENTTYPE']
bookname_list = ['not', 'LIBRARY','CONTACT','GRADUATE','FACILITY','DATE','DAYLENGTH','ITEMLENGTH','PARTIALITEMLENGTH','MEMBERSHIP','PURPOSE','SUBJECT','LOCATION','UNIVERSITY','STUDENTTYPE','PRINTING','TRAINING','LIBRARYMATERIAL','LIBRARYSTAFF','INTERNET','PAYMENT','BOOKNAME','DEPENDENT','TOPIC','AUTHENTICATION']
labeled_features = []

# training_file = open("slots-v2/slot_questions.txt", "rb")
# training_label_file = open("slots-v2/slot_index.txt", "rb")

training_file = open("../chat_history/slot_questions.txt", "rb")
training_label_file = open("../chat_history/slot_index.txt", "rb")

# TODO
testing_file = open("../chat_history/training_questions.txt", "rb")
# output_file = open("book_name_output.txt", "wb")

#****************************************************************building input features



def MEMM_features(trigram, word, post_tri, previous_tag, pre_pos, punct):

    features = {}
#    features['previous4_word'] = four_word
#    features['previous3_word'] = three_word
#    features['previous2_word'] = two_word
#    features['previous_word'] = one_word
	# features['current_word10'] = word
	# features['current_word9'] = word
	# features['current_word8'] = word
#	# features['current_word7'] = word
#    features['current_word6'] =  word
#    features['current_word5'] = word
#    features['current_word4'] = word
#    features['current_word3'] = word
#    features['current_word2'] = word
    features['trigram'] = trigram
#    features['bigram'] = bigram
    features['current_word'] = word
#    features['last_word'] = word_one
#    features['last2_word'] = word_two
	# features['bookname'] = tag
	# puc = '-'.decode("utf-8")
	# some char is outof ASCII
    features['post_tri']=post_tri
	# features['begin_capitalization'] = begin_capitalization # => shound have been captured by upper but not book tag
    features['capitalization'] = word[0].isupper()
    features['previous_tag'] = previous_tag
    features['pre_pos']=pre_pos
    features['punct']=punct
    return features


for x, y in izip(training_file, training_label_file):
    x = x.strip()

    x = x.translate(None, string.punctuation)


    sentenceList = x.split()
    labelList = y.split()
    start_word = "startttt"
    end_word = "endddd"
    not_word = "not"


    for iiii in range(4):
        sentenceList.insert(0,start_word)
        labelList.insert(0,not_word)

    for ii in range(2):
        # sentenceList.insert(0,start_word)
        sentenceList.append(end_word)
        labelList.append(not_word)

    for iii in range(4, len(sentenceList)-2):
        trigram=sentenceList[iii-3]+'_'+sentenceList[iii-2]+'_'+sentenceList[iii-1]
        post_tri=sentenceList[iii+1]+'_'+sentenceList[iii+2]
        pre_pos=nltk.pos_tag(nltk.word_tokenize(sentenceList[iii-1]))
        if (sentenceList[iii+1] in string.punctuation)or (sentenceList[iii+2]in string.punctuation):
            punct=True
        else:
            punct=False
        item = trigram, sentenceList[iii], post_tri, pre_pos[0][1], labelList[iii-1], sentenceList[iii], punct
        labeled_features.append(item)


#****************************************************************training
for xx in range(len(labeled_features)):
    print(labeled_features[xx])

print("****************************************************************")
train_set = [(MEMM_features(trigram, word, post_tri, previous_tag, pre_pos, punct), real_tag) for (trigram, word, post_tri, previous_tag, pre_pos, real_tag, punct) in labeled_features]

for xxx in range(len(train_set)):
    print(train_set[xxx])

print("############")
print("start training", time.strftime('%X %x %Z'))
f = open("MEMM-multi-label-v4.pickle", "wb")
maxent_classifier = MaxentClassifier.train(train_set, algorithm="gis", max_iter=40)
pickle.dump(maxent_classifier , f)
f.close()
print("finish training", time.strftime('%X %x %Z'))

