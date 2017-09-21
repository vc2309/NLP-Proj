import nltk
from nltk.stem.porter import *
from nltk.classify import MaxentClassifier
import pickle
import os,sys
from io import open
import string
import time
from nltk.tag import StanfordNERTagger

def reader():
    f=open("../chat_history/q.txt",'rb')
    file=open("../chat_history/ner_train2.txt",'ab')
    tagger=StanfordNERTagger('/Users/vishnuchopra/Project/stanford-ner-2017-06-09/classifiers/english.conll.4class.distsim.crf.ser.gz', '/Users/vishnuchopra/Project/stanford-ner-2017-06-09/stanford-ner.jar')
    for line in f:
        print(line)
        line=line.strip()
        tags=tagger.tag(line.split())
        for tag in tags:
            nert=tag[1]
            print(nert+' ')
            file.write(nert+' ')
        file.write('\n')
        print('\n')

def decode():
    file=open("../chat_history/ner_train.txt",'rb')
    for line in file:
        line=line.strip()
        x=line.split()
        print(x[0])
        print("**********")
def count():
    with open("../chat_history/ner_train2.txt",'rb') as f2:
        for i,l in enumerate(f2):
            pass
    print(i+1)

def main():
    a=[1,2,3,4,5,6,7]
    b=4
    print(4/float(len(a)-1))
    choice=raw_input("R/D \n")
    if choice=='R': #Select R to generate ner label file
        reader()
    elif choice=='D': #debugging purpose
        decode()
    else :
        count()

if __name__=='__main__':
    main()
