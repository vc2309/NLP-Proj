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
from nltk.tag import StanfordNERTagger

#reload(sys)
#sys.setdefaultencoding('utf8')

print("start program ", time.strftime('%X %x %Z'))

#from itertools import izip
#from itertools import groupby

#bookname_list = ['not', 'LIBRARY', 'GRADUATE', 'FACILITY', 'DATE', 'MEMBERSHIP', 'PURPOSE', 'SUBJECT', 'LOCATION', 'UNIVERSITY', 'STUDENTTYPE']
#bookname_list = ['not', 'LIBRARY','CONTACT','GRADUATE','FACILITY','DATE','DAYLENGTH','ITEMLENGTH','PARTIALITEMLENGTH','MEMBERSHIP','PURPOSE','SUBJECT','LOCATION','UNIVERSITY','STUDENTTYPE','PRINTING','TRAINING','LIBRARYMATERIAL','LIBRARYSTAFF','INTERNET','PAYMENT','BOOKNAME','DEPENDENT','TOPIC','AUTHENTICATION','RELATION']

#UNCOMMENT ABOVE^ TO IMPLEMENT SPECIFIC LABELS

labeled_features = []

#test_file = open("../chat_history/kq.txt", "rb")
#test_label_file = open("../chat_history/ks.txt", "rb")
with open("../chat_history/kq.txt", "r") as f:
    for i, l in enumerate(f):
        pass
filelen=i+1

#########################################
#Comment out this section to remove universal label
bookname_list=['not']
#count unique labels in label training file
with open("../chat_history/ks.txt", "r") as lab_file: #edit file path
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
testing_file = open("../chat_history/kq.txt", "r")
test_labels  = open("../chat_history/ks.txt","r")

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
#****************************************************************training
#for xx in range(len(labeled_features)):
#    print(labeled_features[xx])


print("****************************************************************")
#train_set = [(MEMM_features(bigram, word, post_bi, previous_tag, punct), real_tag)for (bigram, word, post_bi, previous_tag, real_tag,punct) in labeled_features]
#for xxx in range(len(train_set)):
#    print(train_set[xxx])

print("############")
#print("start training", time.strftime('%X %x %Z'))
#f = open("MEMM-multi-label-v46.pickle", "wb")
#maxent_classifier = MaxentClassifier.train(train_set, algorithm="gis", max_iter=40)
#pickle.dump(maxent_classifier , f)
#f.close()
f = open('MEMM-multi-label-vNEW.pickle', 'rb')
maxent_classifier = pickle.load(f,encoding='latin1')


def MEMM(wordList,labList):
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
            backpointer[t][w] = bookname_list[maxPreviousState] # given current tag => what we think is the biggest chance previous tag is ******
        file=open("test_reports.txt",'a')
        file2=open("error_reports.txt",'a')
        print(wordList)
        sublist=wordList[4:len(wordList)-2]
        for word in sublist:
            file.write(word+' ')
        file.write('\n')
        file.write("CURRENT WORD "+wordList[w]+'\n')
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
                else:
                    sublist=wordList[4:len(wordList)-2]
                    for word in sublist:
                        file2.write(word+' ')
                    file2.write('\n')
                    file2.write("current word "+wordList[w-1]+'\n')
                    file2.write("predicted tag :"+bookname_list[maxPreviousState]+"     correct tag: "+labList[w-5]+'\n')
                    file2.write('\n')
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
##    acc=(total/(len(wordList)-6))
##    file.write("ACCURACY= %f "%(acc))
#    print path
    spath=list(zip(path,scores))
    return total,spath


nm=0
total=0
ctr=0
fin=0
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

for line, label in zip(testing_file,test_labels):
    if ctr<=round(filelen):
        line = line
        sentenceList = line.split()
        t_label_list = label.split()
        #nerList=ner.split()
        print("sentenceList", sentenceList)
        print("before MEMM", time.strftime('%X %x %Z'))
        correct,path=MEMM(sentenceList,t_label_list)
        total = total+correct
        fin=fin+correct
        nm=nm+float(len(sentenceList)-7)
        print(nm)
        print(fin)
        perc=(correct/float(len(sentenceList)-7))
        print (" %s CORRECT OUT OF %s " %(correct,len(sentenceList)-7))
        print ("TOKEN PERCENTAGE ACC : %f" %(perc))
        with open("tester.txt",'a') as fi:
            fi.write(line)
            fi.write('\n')
            for item,item2 in path:
                fi.write(item+", ")
                fi.write(str(item2)+" ")
            fi.write('\n')
            fi.write(" %s CORRECT OUT OF %s \n" %(correct,len(sentenceList)-7))
            fi.write("TOKEN PERCENTAGE ACC : %f \n" %(perc))
            fi.write('\n')
            fi.write('\n')
            fi.write('\n')
        print("after MEMM", time.strftime('%X %x %Z'))
        print(total/nm)
    ctr=ctr+1
    print(ctr)
print("FINAL= %f" %(fin/float(nm)))
