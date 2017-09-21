"""
TODO:

add QUERY FEATURE - if word before or 2word before is query term
edit training data

"""

import nltk
from nltk.stem.porter import *
from nltk.classify import MaxentClassifier
import pickle
import os,sys
from io import open
import string
import time
from nltk.tag import StanfordNERTagger
import numpy as np
import linecache
reload(sys)
sys.setdefaultencoding('utf8')

print("start program ", time.strftime('%X %x %Z'))

from itertools import izip
from itertools import groupby

#bookname_list = ['not', 'LIBRARY', 'GRADUATE', 'FACILITY', 'DATE', 'MEMBERSHIP', 'PURPOSE', 'SUBJECT', 'LOCATION', 'UNIVERSITY', 'STUDENTTYPE']
#bookname_list = ['not', 'LIBRARY','CONTACT','GRADUATE','FACILITY','DATE','DAYLENGTH','ITEMLENGTH','PARTIALITEMLENGTH','MEMBERSHIP','PURPOSE','SUBJECT','LOCATION','UNIVERSITY','STUDENTTYPE','PRINTING','TRAINING','LIBRARYMATERIAL','LIBRARYSTAFF','INTERNET','PAYMENT','BOOKNAME','DEPENDENT','TOPIC','AUTHENTICATION','RELATION']

#UNCOMMENT ABOVE^ TO IMPLEMENT SPECIFIC LABELS
query_terms=['related','relation','relating','from','by','written','about','from','for','regarding']
labeled_features = []

training_file = open("../chat_history/q.txt", "rb")
training_label_file = open("../chat_history/s.txt", "rb")
with open("../chat_history/q.txt", "rb") as f:
    for i, l in enumerate(f):
        pass
filelen=i+1

#########################################
#Comment out this section to remove universal label
bookname_list=['not']
#count unique labels in label training file
with open("../chat_history/s.txt", "rb") as lab_file:
    for line in lab_file:
        tags=line.split()
        for tag in tags:
            if tag not in bookname_list:
                bookname_list.append(tag)

print(bookname_list)
print("NO OF LABELS= %s" %(len(bookname_list)))
############################################
print("FILELEN= %s" %(filelen))
# TODO
testing_file = open("../chat_history/q.txt", "rb")
test_labels  = open("../chat_history/s.txt","rb")
#train_ner=open("../chat_history/ner_train2.txt","rb")
#test_ner=open("../chat_history/ner_train2.txt","rb")

def MEMM_features(bigram, word, post_bi, previous_tag,punct,query):
    features = {}
    features['bigram'] = bigram
    features['current_word'] = word
    current_pos=nltk.pos_tag(nltk.word_tokenize(word))
    features['current_pos']=current_pos[0][1]
    features['post_bi']=post_bi
    features['capitalization'] = word[0].isupper()
    features['previous_tag'] = previous_tag
    features['puntctuation']=punct
    features['query']=query
    #features['ner_tag']=nt
    return features

ctr1=0
linenos=[]
    #for x, y in izip(training_file, training_label_file):
for ctr1 in range(int(round(filelen*0.7))):
    lineno=np.random.randint(0,filelen-1)
    linenos.append(lineno)
    x=linecache.getline("../chat_history/q.txt", lineno, module_globals=None)
    y=linecache.getline("../chat_history/s.txt", lineno, module_globals=None)
    if ctr1<=round(filelen*0.7):
        x = x.strip()
    
        x = x.translate(None, string.punctuation)
        
    
        sentenceList = x.split()
        labelList = y.split()
        start_word = "startttt"
        end_word = "endddd"
        not_word = "not"
        #ner_tags=z.split()

        for iiii in range(4):
            sentenceList.insert(0,start_word)
            labelList.insert(0,not_word)
        #   ner_tags.insert(0,'O')
        for ii in range(2):
            # sentenceList.insert(0,start_word)
            sentenceList.append(end_word)
            labelList.append(not_word)
        #   ner_tags.append('O')
        for iii in range(4, len(sentenceList)-2):
            if (sentenceList[iii+1] in string.punctuation)or (sentenceList[iii+2]in string.punctuation):
                punct=1
            else:
                punct=0
            bigram=sentenceList[iii-2]+'_'+sentenceList[iii-1]
            post_bi=sentenceList[iii+1]+'_'+sentenceList[iii+2]
            if (sentenceList[iii-1] in query_terms) or (sentenceList[iii-2] in query_terms):
                query=1
            else:
                query=0
                        
            #  nt=ner_tags[iii]
            item = bigram, sentenceList[iii], post_bi, labelList[iii-1], labelList[iii],punct,query

            labeled_features.append(item)
        ctr1=ctr1+1
    else:
        break
    print item

#****************************************************************training
for xx in range(len(labeled_features)):
    print(labeled_features[xx])


print("****************************************************************")
train_set = [(MEMM_features(bigram, word, post_bi, previous_tag, punct, query), real_tag)for (bigram, word, post_bi, previous_tag, real_tag,punct,query) in labeled_features]
for xxx in range(len(train_set)):
    print(train_set[xxx])

print("############")
print("start training", time.strftime('%X %x %Z'))
f = open("MEMM-multi-label-v50.pickle", "wb")
maxent_classifier = MaxentClassifier.train(train_set, algorithm="gis", max_iter=40)
pickle.dump(maxent_classifier , f)
f.close()
print("finish training", time.strftime('%X %x %Z'))



f = open('MEMM-multi-label-v50.pickle', 'rb')
maxent_classifier = pickle.load(f)


def MEMM(wordList, labList):
    total=0
    start_word = "startttt"
    end_word = "endddd"
    

    for d in range(4):
        wordList.insert(0,start_word)
    #nerList.insert(0,'O')
    for d in range(3):
        wordList.append(end_word)
#   nerList.append('O')
    tRange = len(bookname_list)
    wRange = len(wordList)

    viterbi = [[0 for x in range(300)] for x in range(300)] # store the highest probabilities value
    backpointer = [['' for x in range(300)] for x in range(300)] # store tag that has the highest probabilities value
#   ner_tags=nerList

    for w in range(4, wRange-2):
        maxViterbi = 0
        maxViterbiList = [0] * len(bookname_list)
        maxPreviousState = 0
		# all combination of prev + current tag probabilities => determine what the previous tag is *******
        scores=[0]
        for t in range (tRange): # current tag
            for i in range (tRange): # prev tag
                if (wordList[w+1] in string.punctuation)or (wordList[w+2]in string.punctuation):
                    punct=1
                else:
                    punct=0
                bigram=wordList[w-2]+'_'+wordList[w-1]
                post_bi=wordList[w+1]+'_'+wordList[w+2]
                if (wordList[w-1] in query_terms) or (wordList[w-2] in query_terms):
                    query=1
                else:
                    query=0

                #               nt=ner_tags[w]
                
                if w == 4:
                    probability = maxent_classifier.prob_classify(MEMM_features(bigram,wordList[w],post_bi,bookname_list[0],punct,query))
                    posterior = float(probability.prob(bookname_list[t])) + 1 # prob of current tag with prev tag as b_l[i] (transition probability)

                    if posterior > maxViterbiList[t]:
                        maxViterbiList[t] = posterior #set maxViterbi to find the prev tag which will give max probability for the current tag for current word

                    if posterior > maxViterbi:
                        maxViterbi = posterior
                        maxPreviousState = i

                if w > 4:
                    probability = maxent_classifier.prob_classify(MEMM_features(bigram,wordList[w],post_bi,bookname_list[i],punct,query))
                    posterior = float(probability.prob(bookname_list[t])) + 1 #probability of current tag at bookname_list[t] with prev tag set as bookname_list[i]

                    if float(viterbi[i][w-1]) * posterior > maxViterbi:
                        maxViterbi = float(viterbi[i][w-1]) * posterior
                        maxPreviousState = i


                    if float(viterbi[i][w-1]) * posterior > maxViterbiList[t]:
                        maxViterbiList[t] = float(viterbi[i][w-1]) * posterior


            scores[t]=maxViterbiList[t]
            scores.append(0)
            viterbi[t][w] = maxViterbiList[t] / 2 # prevent int explosion
            backpointer[t][w] = bookname_list[maxPreviousState] # given current tag => what we think is the biggest chance previous tag is ******
        file=open("testing4.txt",'ab')
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
        if w>4:
            file.write("CORRECT TAG= "+labList[w-5]+" PREDICTED PREV TAG= "+bookname_list[maxPreviousState]+"\n")
            if labList[w-5]!='not':
                if bookname_list[maxPreviousState]!='not':
                    total=total+1
            elif labList[w-5]=='not' and bookname_list[maxPreviousState]=='not':
                total=total+1





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
    scores[0]=scores[0]/2

#    print(1)
#    print(wordList[4])
#    print(maxPrevTag)
#    print("\n")
#    acc=(total/(len(wordList)-6))
#    file.write("ACCURACY= %f "%(acc))
#    print path
#return total
    print(list(zip(path,scores)))
    return(list(zip(path,scores)))

nm=0
total=0
ctr=0
fin=0
for ctr in range(int(round(0.3*filelen))):
    lineno=np.random.randint(0,filelen)
    while lineno in linenos:
        lineno=np.random.randint(0,filelen)
    line=linecache.getline("../chat_history/q.txt", lineno, module_globals=None)
    label=linecache.getline("../chat_history/s.txt", lineno, module_globals=None)
    line = line.translate(None, string.punctuation)
    sentenceList = line.split()
    t_label_list = label.split()
    MEMM(sentenceList,t_label_list)
