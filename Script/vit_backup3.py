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

reload(sys)
sys.setdefaultencoding('utf8')

print("start program ", time.strftime('%X %x %Z'))

from itertools import izip
from itertools import groupby

#bookname_list = ['not', 'LIBRARY', 'GRADUATE', 'FACILITY', 'DATE', 'MEMBERSHIP', 'PURPOSE', 'SUBJECT', 'LOCATION', 'UNIVERSITY', 'STUDENTTYPE']
bookname_list = ['not', 'LIBRARY','CONTACT','GRADUATE','FACILITY','DATE','DAYLENGTH','ITEMLENGTH','PARTIALITEMLENGTH','MEMBERSHIP','PURPOSE','SUBJECT','LOCATION','UNIVERSITY','STUDENTTYPE','PRINTING','TRAINING','LIBRARYMATERIAL','LIBRARYSTAFF','INTERNET','PAYMENT','BOOKNAME','DEPENDENT','TOPIC','AUTHENTICATION','RELATION']

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


f = open('MEMM-multi-label-v46.pickle', 'rb')
maxent_classifier = pickle.load(f)

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
        # all combination of prev + current tag probabilities => determine what the previous tag is *******
        for t in range (tRange): # current tag
            for i in range (tRange): # prev tag
                if (wordList[w+1] in string.punctuation)or (wordList[w+2]in string.punctuation):
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
                
#                    print(wordList[w-4])
#                    print(wordList[w-3])
#                    print(wordList[w-2])
#                    print(wordList[w-1])
#                    print("CURR WORD : ", wordList[w])
#                    print(wordList[w+1])
#                    print(wordList[w+2])
#                    print("prev tag: ", bookname_list[0])
#                    print("curr tag: ", bookname_list[t])
#                    print "current prob ", posterior
#                    print "max prob ", maxViterbi
#                    print "\n"

                    
                    if posterior > maxViterbi:
                        maxViterbi = posterior
                        maxPreviousState = i
    
    
    
    
                if w > 4:
                    probability = maxent_classifier.prob_classify(MEMM_features(bigram,wordList[w],post_bi,bookname_list[i],punct))
                    posterior = float(probability.prob(bookname_list[t])) + 1 #probability of current tag at bookname_list[t] with prev tag set as bookname_list[i]
                    
#                    print(wordList[w-4])
#                    print(wordList[w-3])
#                    print(wordList[w-2])
#                    print(wordList[w-1])
#                    print "CURR WORD : ", wordList[w]
#                    print(wordList[w+1])
#                    print(wordList[w+2])
#                    print "prev tag: " + bookname_list[i]
#                    print "curr tag: " + bookname_list[t]
#                    print "curr tag --", bookname_list[t], "-- prob: ", posterior
#                    print "prev tag --", bookname_list[i], "-- highest prob : ", float(viterbi[i][w-1])
#                    print "combined prev + curr prob ", float(viterbi[i][w-1]) * posterior #probability of previous word's probability of tag bookname_list[i] multiplied by the current words probability of tag bookname_list[t]
#                    print "max prob ", maxViterbi
#                    print "--",bookname_list[t],"-- curr tag: highest value to be stored in viterbi at this time step (before) : ", maxViterbiList[t]

                    
                    if float(viterbi[i][w-1]) * posterior > maxViterbi:
                        maxViterbi = float(viterbi[i][w-1]) * posterior
                        maxPreviousState = i
                    
                    
                    if float(viterbi[i][w-1]) * posterior > maxViterbiList[t]:
                        maxViterbiList[t] = float(viterbi[i][w-1]) * posterior
#                        print("value updated")
#                    
#                    print "--",bookname_list[t],"-- curr tag: highest value to be stored in viterbi at this time step (before) : ", maxViterbiList[t]
#                    print ("\n")

        
            viterbi[t][w] = maxViterbiList[t] / 2 # prevent int explosion
            backpointer[t][w] = bookname_list[maxPreviousState] # given current tag => what we think is the biggest chance previous tag is ******
#        file=open("testing.txt",'ab')
#        sublist=wordList[4:len(wordList)-2]
#        for word in sublist:
#            file.write(word+' ')
#        file.write('\n')
#        file.write("CURRxENT WORD "+wordList[w]+'\n')
#        print("CURRENT WORD", wordList[w])
#        for t in range(tRange):
#            print(bookname_list[t]+" prob - "+str(viterbi[t][w]))
#            file.write(bookname_list[t]+" prob - "+str(viterbi[t][w]))
#            print("backpointing at tag - "+backpointer[t][w])
#            file.write(" backpointing at tag - "+backpointer[t][w]+'\n')
#    if w>4:
#        file.write("CORRECT TAG= "+labList[w-5]+" PREDICTED PREV TAG= "+bookname_list[maxPreviousState]+"\n")
#            if labList[w-5]!='not':
#                if bookname_list[maxPreviousState]!='not':
#                    total=total+1
#        elif labList[w-5]=='not' and bookname_list[maxPreviousState]=='not':
#            total=total+1





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

#for line, label in zip(testing_file,test_labels):
#    if ctr>=round(0.7*filelen):
#        line = line.translate(None, string.punctuation)
#        sentenceList = line.split()
#        t_label_list = label.split()
#        #nerList=ner.split()
#        print("sentenceList", sentenceList)
#        print("before MEMM", time.strftime('%X %x %Z'))
#        correct=MEMM(sentenceList,t_label_list)
#        total = total+correct
#        fin=fin+correct
#        nm=nm+float(len(sentenceList)-6)
#        print(nm)
#        print(fin)
#        perc=(correct/float(len(sentenceList)-6))
#        print (" %s CORRECT OUT OF %s " %(correct,len(sentenceList)-6))
#        print ("TOKEN PERCENTAGE ACC : %f" %(perc))
#        with open("testing.txt",'ab') as fi:
#            fi.write(" %s CORRECT OUT OF %s \n" %(correct,len(sentenceList)-6))
#            fi.write("TOKEN PERCENTAGE ACC : %f \n" %(perc))
#        print("after MEMM", time.strftime('%X %x %Z'))
#        print(total/nm)
#    ctr=ctr+1
#    print(ctr)
#print("FINAL= %f" %(fin/float(nm)))
qs=int(raw_input("how many queries\n"))
for i in range(qs):
    line=raw_input("enter\n")
    tb=time.strftime('%X %x %Z')
    sentenceList = line.split()
    MEMM(sentenceList)
    print("start time ",tb, " time after ", time.strftime('%X %x %Z'))
