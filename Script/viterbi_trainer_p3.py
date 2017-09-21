import nltk
from nltk.stem.porter import *
from nltk.classify import MaxentClassifier
import pickle
import os,sys
from io import open
import string
import time
from nltk.tag import StanfordNERTagger

#reload(sys)
#sys.setdefaultencoding('utf8')

print("start program ", time.strftime('%X %x %Z'))

#from itertools import izip
#from itertools import groupby

#bookname_list = ['not', 'LIBRARY', 'GRADUATE', 'FACILITY', 'DATE', 'MEMBERSHIP', 'PURPOSE', 'SUBJECT', 'LOCATION', 'UNIVERSITY', 'STUDENTTYPE']
#bookname_list = ['not', 'LIBRARY','CONTACT','GRADUATE','FACILITY','DATE','DAYLENGTH','ITEMLENGTH','PARTIALITEMLENGTH','MEMBERSHIP','PURPOSE','SUBJECT','LOCATION','UNIVERSITY','STUDENTTYPE','PRINTING','TRAINING','LIBRARYMATERIAL','LIBRARYSTAFF','INTERNET','PAYMENT','BOOKNAME','DEPENDENT','TOPIC','AUTHENTICATION','RELATION']

#UNCOMMENT ABOVE^ TO IMPLEMENT SPECIFIC LABELS
#query_terms=['related','relation','relating','from','by','written','about','from','for','regarding']

#with open("../chat_history/q.txt", "rb") as f:
#    for i, l in enumerate(f):
#        pass
#filelen=i+1

#########################################
#Comment out this section to remove universal label
#bookname_list=['not']
#count unique labels in label training file
#with open("../chat_history/s.txt", "rb") as lab_file:
#    for line in lab_file:
#        tags=line.split()
#        for tag in tags:
#            if tag not in bookname_list:
#                bookname_list.append(tag)

bookname_list=['not']
with open("../chat_history/ks2.txt", "r") as lab_file:
    for line in lab_file:
        tags=line.split()
        for tag in tags:
            if tag not in bookname_list:
                bookname_list.append(tag)

print(bookname_list)
print("NO OF LABELS= %s" %(len(bookname_list)))
def MEMM_features(bigram, word, post_bi, previous_tag,punct):
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


f = open('MEMM-multi-label-vNEW3.pickle', 'rb') #EDIT FILE LOCATION
maxent_classifier = pickle.load(f,encoding='latin1') #REMOVE ENCODING WHEN USING PYTHON2

def MEMM(wordList):
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
        scores=[0]
        # all combination of prev + current tag probabilities => determine what the previous tag is *******
        for t in range (tRange): # current tag
            for i in range (tRange): # prev tag
                if (wordList[w+1] in string.punctuation) or (wordList[w+2]in string.punctuation):
                    punct=1
                else:
                    punct=0
                bigram=wordList[w-2]+'_'+wordList[w-1]
                post_bi=wordList[w+1]+'_'+wordList[w+2]

                #               nt=ner_tags[w]

                if w == 4:
                    probability = maxent_classifier.prob_classify(MEMM_features(bigram,wordList[w],post_bi,bookname_list[0],punct))
                    posterior = float(probability.prob(bookname_list[t])) + 1 # prob of current tag with prev tag as b_l[i] (transition probability)

                    if posterior > maxViterbiList[t]:
                        maxViterbiList[t] = posterior #set maxViterbi to find the prev tag which will give max probability for the current tag for current word


                    if posterior > maxViterbi:
                        maxViterbi = posterior
                        maxPreviousState = i




                if w > 4:
                    probability = maxent_classifier.prob_classify(MEMM_features(bigram,wordList[w],post_bi,bookname_list[i],punct))
                    posterior = float(probability.prob(bookname_list[t])) + 1 #probability of current tag at bookname_list[t] with prev tag set as bookname_list[i]



                    if float(viterbi[i][w-1]) * posterior > maxViterbi:
                        maxViterbi = float(viterbi[i][w-1]) * posterior
                        maxPreviousState = i


                    if float(viterbi[i][w-1]) * posterior > maxViterbiList[t]:
                        maxViterbiList[t] = float(viterbi[i][w-1]) * posterior
            scores[t]=maxViterbiList[t]
            scores.append(0)
            viterbi[t][w] = maxViterbiList[t] / 2 # prevent int explosion
            backpointer[t][w] = bookname_list[maxPreviousState] # given current tag => what we think is the

    maxPrevTag = bookname_list[maxPreviousState]
    path = [maxPrevTag]


    # from the actual last word in sentence
    for i in range(wRange-4, 4, -1):
        maxPrevTag = backpointer[maxPreviousState][i]
        path.insert(0, maxPrevTag)
        maxPreviousState = bookname_list.index(maxPrevTag)
    scores[0]=scores[0]/2
    print(list(zip(path,scores)))
    # return(list(zip(path,scores)))
qs=int(input("how many queries\n"))
for i in range(qs):
   line=input("enter\n")
   tb=time.strftime('%X %x %Z')
   sentenceList = line.split()
   MEMM(sentenceList)
   print("start time ",tb, " time after ", time.strftime('%X %x %Z'))
