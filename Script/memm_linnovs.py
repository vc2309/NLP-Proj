#!/usr/bin/python
# -*- coding: utf-8 -*-

import nltk
from nltk.stem.porter import *
from nltk.classify import MaxentClassifier
import pickle
import os,sys
import re
from io import open
import string
import time
#from nltk.tag import StanfordNERTagger
from chatbot.helper import debug

# reload(sys)
# sys.setdefaultencoding('utf8')

 debug("start program ", time.strftime('%X %x %Z'))

# from itertools import zip
from itertools import groupby

#bookname_list = ['not', 'LIBRARY', 'GRADUATE', 'FACILITY', 'DATE', 'MEMBERSHIP', 'PURPOSE', 'SUBJECT', 'LOCATION', 'UNIVERSITY', 'STUDENTTYPE']
labeled_features = []


# testing_file = open("title_questsions_test.txt", "r")
# output_file = open("book_name_output.txt", "wb")

#****************************************************************building input features

#Comment out this section to remove universal label
bookname_list=['not']
#tagger=StanfordNERTagger('/Users/vishnuchopra/Project/stanford-ner-2017-06-09/classifiers/english.conll.4class.distsim.crf.ser.gz', '/Users/vishnuchopra/Project/stanford-ner-2017-06-09/stanford-ner.jar') #edit path to give correct location of training file and jar file for NERTagger
#count unique labels in label training file
with open("../chat_history/s.txt", "rb") as lab_file: #Edit to correct labeling file
    for line in lab_file:
        tags=line.split()
        for tag in tags:
            if tag not in bookname_list:
                bookname_list.append(tag)

#print(bookname_list)
#print("NO OF LABELS= %s" %(len(bookname_list)))
############################################


def GeneralSlot_MEMM_features(bigram, word, post_bi, previous_tag,punct):
    features = {}
    features['bigram'] = bigram
    features['current_word'] = word
    current_pos=nltk.pos_tag(nltk.word_tokenize(word))
    features['current_pos']=current_pos[0][1]
    features['post_bi']=post_bi
    features['capitalization'] = word[0].isupper()
    features['previous_tag'] = previous_tag
    features['puntctuation']=punct
    #features['ner_tag']=nt
    return features

with open('resources/my_classifier_train.pickle', 'rb') as f:
	general_maxent_classifier = pickle.load(f)


def GeneralSlot_MEMM(wordList):

    start_word = "startttt"
    end_word = "endddd"
    for d in range(4):
        wordList.insert(0,start_word)
    for d in range(3):
        wordList.append(end_word)

    tRange = len(bookname_list)
    wRange = len(wordList)
    #nerList = tagger.tag(wordList)
    viterbi = [[0 for x in range(300)] for x in range(300)] # store the highest probabilities value
    backpointer = [['' for x in range(300)] for x in range(300)] # store tag that has the highest probabilities value

    for w in range(4, wRange-2):
        maxViterbi = 0
        maxViterbiList = [0,0,0,0,0,0,0,0,0,0,0,0]
        maxPreviousState = 0
        bigram=wordList[w-2]+'_'+wordList[w-1]
        post_bi=wordList[w+1]+'_'+wordList[w+2]
        #nt=nerList[w]

        for t in range (tRange):
			for i in range (tRange):


				if w == 4:
					probability = general_maxent_classifier.prob_classify(GeneralSlot_MEMM_features(bigram,wordList[w],post_bi,bookname_list[0],punct))
					posterior = float(probability.prob(bookname_list[t])) + 1

					if posterior > maxViterbiList[t]:
						maxViterbiList[t] = posterior


					if posterior > maxViterbi:
						maxViterbi = posterior
						maxPreviousState = i

				if w > 4:
					probability = general_maxent_classifier.prob_classify(GeneralSlot_MEMM_features(bigram,wordList[w],post_bi,bookname_list[i],punct))
					posterior = float(probability.prob(bookname_list[t])) + 1

					if float(viterbi[i][w-1]) * posterior > maxViterbi:
						maxViterbi = float(viterbi[i][w-1]) * posterior
						maxPreviousState = i

					if float(viterbi[i][w-1]) * posterior > maxViterbiList[t]:
						maxViterbiList[t] = float(viterbi[i][w-1]) * posterior

			viterbi[t][w] = maxViterbiList[t] / 2
			backpointer[t][w] = bookname_list[maxPreviousState]

	maxPrevTag = bookname_list[maxPreviousState]
	path = [maxPrevTag]

	for i in range(wRange-4, 4, -1):
		# debug(i-3)
		# debug(wordList[i])
		# debug(maxPrevTag)
		# debug("\n")
		maxPrevTag = backpointer[maxPreviousState][i]
		path.insert(0, maxPrevTag)
		maxPreviousState = bookname_list.index(maxPrevTag)

	debug(1)
	debug(wordList[4])
	debug(maxPrevTag)
	debug("\n")

	return path




# for line in testing_file:
# 	# line = line.translate(None, string.punctuation)
# 	line = re.sub(r'[^\w\s]', '', line)
# 	sentenceList = line.split()

# 	debug("before MEMM", time.strftime('%X %x %Z'))
# 	path = MEMM(sentenceList)
# 	debug("after MEMM", time.strftime('%X %x %Z'))
# 	debug (path)

bookname_list = ['not', 'book']
labeled_features = []

with open('resources/my_classifier_train.pickle', 'rb') as f:
	book_maxent_classifier = pickle.load(f)


def BookSlot_MEMM_features(bigram, word, post_bi, previous_tag,punct):
    features = {}
    features['bigram'] = bigram
    features['current_word'] = word
    current_pos=nltk.pos_tag(nltk.word_tokenize(word))
    features['current_pos']=current_pos[0][1]
    features['post_bi']=post_bi
    features['capitalization'] = word[0].isupper()
    features['previous_tag'] = previous_tag
    features['puntctuation']=punct
    #features['ner_tag']=nt
    return features

def BookSlot_MEMM(wordList):

    start_word = "startttt"
    end_word = "endddd"

    for d in range(2):
        wordList.insert(0,start_word)
    for d in range(3):
        wordList.append(end_word)

    tRange = len(bookname_list)
    wRange = len(wordList)
    #nerList = tagger.tag(wordList)
    viterbi = [[0 for x in range(300)] for x in range(300)]
    backpointer = [['' for x in range(300)] for x in range(300)]

    for w in range(2, wRange-2):
        maxViterbi = 0
        maxViterbiList = [0,0,0]
        maxPreviousState = 0
        bigram=wordList[w-2]+'_'+wordList[w-1]
        post_bi=wordList[w+1]+'_'+wordList[w+2]
        #nt=nerList[w]
        for t in range (tRange):
            for i in range (tRange):
                if w == 2:
					probability = book_maxent_classifier.prob_classify(BookSlot_MEMM_features(bigram,wordList[w],post_bi,bookname_list[0],punct))
					posterior = float(probability.prob(bookname_list[t])) + 1

					if posterior > maxViterbiList[t]:
						maxViterbiList[t] = posterior

					if posterior > maxViterbi:
						maxViterbi = posterior
						maxPreviousState = i

                if w > 2:
					probability = book_maxent_classifier.prob_classify(BookSlot_MEMM_features(bigram,wordList[w],post_bi,bookname_list[i],punct))
					posterior = float(probability.prob(bookname_list[t])) + 1
					maxViterbiList[t] = posterior

					if float(viterbi[i][w-1]) * posterior > maxViterbi:
						maxViterbi = float(viterbi[i][w-1]) * posterior
						maxPreviousState = i


					if float(viterbi[i][w-1]) * posterior > maxViterbiList[t]:
						maxViterbiList[t] = float(viterbi[i][w-1]) * posterior

            viterbi[t][w] = maxViterbiList[t] / 2
            backpointer[t][w] = bookname_list[maxPreviousState]

	maxPrevTag = bookname_list[maxPreviousState]
	path = [maxPrevTag]

	for i in range(wRange-4, 2, -1):
		# debug(i)
		# debug(wordList[i])
		# debug(maxPrevTag)
		# debug("\n")
		maxPrevTag = backpointer[maxPreviousState][i]
		path.insert(0, maxPrevTag)
		maxPreviousState = bookname_list.index(maxPrevTag)

	return path

