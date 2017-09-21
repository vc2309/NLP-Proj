"""
TODO:

creates dataset => find all slots for hard code slots

add punutation
add context free grammer 
add pos tag 

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
testing_file = open("../chat_history/slot_questions.txt", "rb")
# output_file = open("book_name_output.txt", "wb")

#****************************************************************building input features



def MEMM_features(bigram, word, post_bi, previous_tag,punct):
    features = {}
#    features['previous4_word'] = four_word
#    features['previous3_word'] = three_word
#    features['previous2_word'] = two_word
#    features['previous_word'] = one_word
    features['bigram'] = bigram
        # features['current_word10'] = word
    # features['current_word9'] = word
    # features['current_word8'] = word
    # features['current_word7'] = word
#   features['current_word6'] =  word
#   features['current_word5'] = word
#   features['current_word4'] = word
#   features['current_word3'] = word
#   features['current_word2'] = word
    features['current_word'] = word
    current_pos=nltk.pos_tag(nltk.word_tokenize(word))
    features['current_pos']=current_pos[0][1]
#    features['last_word'] = word_one
#    features['last2_word'] = word_two
    features['post_bi']=post_bi
	# features['bookname'] = tag
	# puc = '-'.decode("utf-8")
	# some char is outof ASCII

	# features['begin_capitalization'] = begin_capitalization # => shound have been captured by upper but not book tag
    features['capitalization'] = word[0].isupper()
    features['previous_tag'] = previous_tag
    features['puntctuation']=punct

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
        if (sentenceList[iii+1] in string.punctuation)or (sentenceList[iii+2]in string.punctuation):
            punct=1
        else:
            punct=0
        bigram=sentenceList[iii-2]+'_'+sentenceList[iii-1]
        post_bi=sentenceList[iii+1]+'_'+sentenceList[iii+2]
        item = bigram, sentenceList[iii], post_bi, labelList[iii-1], labelList[iii],punct

        labeled_features.append(item)


#****************************************************************training
for xx in range(len(labeled_features)):
    print(labeled_features[xx])


print("****************************************************************")
train_set = [(MEMM_features(bigram, word, post_bi, previous_tag, punct), real_tag)for (bigram, word, post_bi, previous_tag, real_tag,punct) in labeled_features]

for xxx in range(len(train_set)):
    print(train_set[xxx])

print("############")
print("start training", time.strftime('%X %x %Z'))
f = open("MEMM-multi-label-v44.pickle", "wb")
maxent_classifier = MaxentClassifier.train(train_set, algorithm="gis", max_iter=40)
pickle.dump(maxent_classifier , f)
f.close()
print("finish training", time.strftime('%X %x %Z'))



f = open('MEMM-multi-label-v4.pickle', 'rb')
maxent_classifier = pickle.load(f)
#test_prob=maxent_classifier({'previous_tag': 'not', 'capitalization': True, 'current_pos': 'NNP', 'last2_word': '', 'last_word': 'library', 'previous2_word': 'will', 'current_word': 'law', 'previous_word': 'the', 'previous3_word': 'When', 'previous4_word': 'startttt'}

def MEMM(wordList):

    start_word = "startttt"
    end_word = "endddd"


    for d in range(4):
        wordList.insert(0,start_word)
    for d in range(3):
        wordList.append(end_word)

    tRange = len(bookname_list)
    wRange = len(wordList)

    viterbi = [[0 for x in range(300)] for x in range(300)] # store the highest probabilities value
    backpointer = [['' for x in range(300)] for x in range(300)] # store tag that has the highest probabilities value


    for w in range(4, wRange-2):
        maxViterbi = 0
        maxViterbiList = [0] * len(bookname_list)
        maxPreviousState = 0
		# all combination of prev + current tag probabilities => determine what the previous tag is *******
        for t in range (tRange): # current tag
            for i in range (tRange): # prev tag
                if (wordList[w+1] in string.punctuation)or (wordList[w+2]in string.punctuation):
                    punct=1
                else:
                    punct=0
                bigram=wordList[w-2]+'_'+wordList[w-1]
                post_bi=wordList[w+1]+'_'+wordList[w+2]
                # print MEMM_features(wordList[w-2],wordList[w-1],wordList[w],wordList[w+1],wordList[w+2],bookname_list[1])
                if w == 4:
                    probability = maxent_classifier.prob_classify(MEMM_features(bigram,wordList[w],post_bi,bookname_list[0],punct))
                    posterior = float(probability.prob(bookname_list[t])) + 1 # prob of current tag with prev tag as b_l[i] (transition probability)

                    if posterior > maxViterbiList[t]:
                        maxViterbiList[t] = posterior #set maxViterbi to find the prev tag which will give max probability for the current tag for current word

                    print(wordList[w-4])
                    print(wordList[w-3])
                    print(wordList[w-2])
                    print(wordList[w-1])
                    print("CURR WORD : ", wordList[w])
                    print(wordList[w+1])
                    print(wordList[w+2])
                    print("prev tag: ", bookname_list[0])
                    print("curr tag: ", bookname_list[t])
                    print "current prob ", posterior
                    print "max prob ", maxViterbi
                    print "\n"



                    # viterbi[t][w] = posterior
                    if posterior > maxViterbi:
                        maxViterbi = posterior
                        maxPreviousState = i
                    # print w
                    # print t
                    # print float(viterbi[t][w])



                if w > 4:
                    probability = maxent_classifier.prob_classify(MEMM_features(bigram,wordList[w],post_bi,bookname_list[i],punct))
                    posterior = float(probability.prob(bookname_list[t])) + 1 #probability of current tag at bookname_list[t] with prev tag set as bookname_list[i]
                    # maxViterbiList[t] = posterior
                    # print float(viterbi[t][w])


                    # if t == 0:
                    #     maxViterbi = float(viterbi[i][w-1]) * posterior
                    #     # print(float(viterbi[i][w-1]))
                    #     # print(posterior)
                    #     print maxViterbi
                    #     maxPreviousState = 0
                    print(wordList[w-4])
                    print(wordList[w-3])
                    print(wordList[w-2])
                    print(wordList[w-1])
                    print "CURR WORD : ", wordList[w]
                    print(wordList[w+1])
                    print(wordList[w+2])
                    print "prev tag: " + bookname_list[i]
                    print "curr tag: " + bookname_list[t]
                    print "curr tag --", bookname_list[t], "-- prob: ", posterior
                    print "prev tag --", bookname_list[i], "-- highest prob : ", float(viterbi[i][w-1])
                    print "combined prev + curr prob ", float(viterbi[i][w-1]) * posterior #probability of previous word's probability of tag bookname_list[i] multiplied by the current words probability of tag bookname_list[t]
                    print "max prob ", maxViterbi
                    print "--",bookname_list[t],"-- curr tag: highest value to be stored in viterbi at this time step (before) : ", maxViterbiList[t]


                    if float(viterbi[i][w-1]) * posterior > maxViterbi:
                        maxViterbi = float(viterbi[i][w-1]) * posterior
                        maxPreviousState = i

                        # print w
                        # print maxPreviousState

                    if float(viterbi[i][w-1]) * posterior > maxViterbiList[t]:
                        maxViterbiList[t] = float(viterbi[i][w-1]) * posterior
                        print("value updated")

                    print "--",bookname_list[t],"-- curr tag: highest value to be stored in viterbi at this time step (before) : ", maxViterbiList[t]
                    print ("\n")


            viterbi[t][w] = maxViterbiList[t] / 2 # prevent int explosion
            backpointer[t][w] = bookname_list[maxPreviousState] # given current tag => what we think is the biggest chance previous tag is ******
        file=open("log4.txt",'ab')
        print(wordList)
        sublist=wordList[4:len(wordList)-2]
        for word in sublist:
            file.write(word+' ')
        file.write('\n')
        file.write("CURRxENT WORD "+wordList[w]+'\n')
        print("CURRENT WORD", wordList[w])
        for t in range(tRange):
            print(bookname_list[t]+" prob - "+str(viterbi[t][w]))
            file.write(bookname_list[t]+" prob - "+str(viterbi[t][w]))
            print("backpointing at tag - "+backpointer[t][w])
            file.write(" backpointing at tag - "+backpointer[t][w]+'\n')

    # for w in range(2, wRange-2):
    #     print("\n")
    #     print("word : ", wordList[w])
    #     for t in range(tRange):
    #         print "tag : ", bookname_list[t]
    #         print "max previous tag : ", backpointer[t][w]
    #         print "max previous tag's combined prob : ", viterbi[t][w]


    # return POS tag path => only maxPreviousState is number, bookname_list or backpointer give you the actual word

	# endddd's maxPrevTag
	maxPrevTag = bookname_list[maxPreviousState]
	path = [maxPrevTag]


    # from the actual last word in sentence
    for i in range(wRange-4, 4, -1):
        print(i-3)
        print(wordList[i])
        print(maxPrevTag)
        print("\n")
        maxPrevTag = backpointer[maxPreviousState][i]
        path.insert(0, maxPrevTag)
        maxPreviousState = bookname_list.index(maxPrevTag)

    print(1)
    print(wordList[4])
    print(maxPrevTag)
    print("\n")

    print path
    # print(wordList[1])
    # print(wordList[0])

	# i = 0
	# while i < (wRange -1):
	# 	pathReverse.append(backpointer[bookname_list.index(maxPreviousTag)][wRange - i])
	# 	maxPreviousTag = backpointer[bookname_list.index(maxPreviousTag)][wRange - i]
	# 	i = i + 1
    #
	# #reverse the path to make it correct
	# index = len(pathReverse)
	# path = []
	# while index >= 1 :
	# 	path.append(pathReverse[index - 1])
	# 	index = index -1
	# return path

#def print_probs(wordList):
#    start_word = "startttt"
#    end_word = "endddd"
#    
#    
#    for d in range(4):
#        wordList.insert(0,start_word)
#    for d in range(3):
#        wordList.append(end_word)
#
#    for w in range(4, len(wordList)-2):
#        i=0
#        bigram=wordList[w-2]+'_'+wordList[w-1]
#        bigram=bigram.lower()
#        post_tri=wordList[w+1]+'_'+wordList[w+2]
#        post_tri=post_tri.lower()
#        pre_pos1=nltk.pos_tag(nltk.word_tokenize(wordList[w-2]))
#        pre_pos2=nltk.pos_tag(nltk.word_tokenize(wordList[w-1]))
#        pre_pos=pre_pos1[0][1]+'_'+pre_pos2[0][1]
#        if (wordList[w+1] in string.punctuation)or (wordList[w+2]in string.punctuation):
#            punct=1
#        else:
#            punct=0
#        
#        if w == 4:
#            probability = maxent_classifier.prob_classify(MEMM_features(bigram,wordList[w],post_tri,bookname_list[0],pre_pos,punct,))
#            prev_tag=str(probability.max())
#        
#        elif w > 4:
#            probability = maxent_classifier.prob_classify(MEMM_features(bigram,wordList[w],post_tri,prev_tag,pre_pos,punct))
#            prev_tag=str(probability.max())
#    
#        print("Word:-", wordList[w])
#        prob_mat=[(bookname_list[t],float(probability.prob(bookname_list[t]))) for t in range(len(bookname_list))]
#        #print(prob_mat)
#    print("max- "+str(probability.max())+" : ",probability.prob(str(probability.max()))  )





for line in testing_file:
    line = line.translate(None, string.punctuation)
    sentenceList = line.split()
    print("sentenceList", sentenceList)
    # start_word = "startttt"
    # end_word = "endddd"
    #
    # for ii in range(2):
    #     sentenceList.insert(0,start_word)
    #     sentenceList.append(end_word)
#print_probs(se)
    print(sentenceList)
    print(len(sentenceList))
    print("before MEMM", time.strftime('%X %x %Z'))
    path = MEMM(sentenceList)
    print("after MEMM", time.strftime('%X %x %Z'))
    print path

