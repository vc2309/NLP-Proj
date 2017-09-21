#!/usr/bin/python
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pycorenlp import StanfordCoreNLP
#from question_identifier import language_processor
import re
import nltk
import pickle
import linecache
import sys
# from testing_x import pre_proc
import pandas as pd
from extract import get_qs
import numpy as np
from collections import Counter
from training_new import master
from nlp_consol import new_feature_engineering_test

tp=0
fn=0
all_ints={}
# # def new_feature_engineering(question):   
#     feature_file=open("models/original/all_features_lower-new_x1.txt",'r')
#     all_features = [i[:len(i)-1] for i in feature_file.readlines()]
#     feature_matrix = []
#     features=[]
#     spam=['i','it','know','what','is','please','pls','ask','tell','me','have']
#     previous_governor_word=''
#     qs=nlp.annotate(question, properties={
#             'annotators': 'tokenize,depparse',
#             'outputFormat': 'json'}
#         ).get('sentences')
#     for q in qs:
#         print(q['basicDependencies'])
#         # print("yeah")
#         splited_line = [i['word'] for i in q['tokens']]
#         print(' '.join(splited_line)) # the q itself
#         for dep_dict in q['basicDependencies']:
#             dependent_word = dep_dict['dependentGloss'].lower()
#             tag = dep_dict['dep']
#             governor_word = dep_dict['governorGloss'].lower()
#             if tag in ('nn', 'neg', 'amod', 'acl:relcl', 'nsubjpass'):
#                 combined_word = dependent_word + '_' + governor_word
#                 features.append(combined_word)
#             if tag=='advmod':
#                     if dependent_word == "where" or dependent_word=="how":
#                         combined_word = dependent_word + '_' + governor_word
#                         features.append(combined_word)
#                         # print(combined_word)
#             if(tag == 'dobj' or tag=='nsubj'):
#                 dobj_combined_word = governor_word + '_' + dependent_word
#                 features.append(dobj_combined_word)
#                 features.append(governor_word)
#                 # features.append(dependent_word)
#             if tag in ('tmod','nmod:tmod'):
#                 tmod_combined=governor_word+'_'+dependent_word
#                 features.append(tmod_combined)
#                 features.append(dependent_word)

#             if tag=='nummod':
#                 nummod_combined=governor_word+'_'+dependent_word
#                 features.append(nummod_combined)
#                 features.append(governor_word)
#                 features.append(dependent_word)
#             if(tag == 'nmod'):
#                 nmod_combined_word = governor_word + '_' + dependent_word
#                 reverse_nmod_combined_word = dependent_word + '_' + governor_word
#                 features.append(nmod_combined_word)


#             if(tag == 'compound'):
#                     # combining multiple compound
#                     if(previous_governor_word == governor_word):
#                         # append the current word first
#                         current_compound_combined_word = dependent_word + '_' + governor_word
#                         features.append(current_compound_combined_word)

#                         all_words = compound_combined_word.split("_")
#                         all_words.insert((len(all_words)-1), dependent_word)
#                         compound_combined_word = "_".join(all_words)
#                         previous_governor_word = governor_word 
#                     else:
#                         compound_combined_word = dependent_word + '_' + governor_word                    
#                         previous_governor_word = governor_word
#                     features.append(compound_combined_word)
#             if(tag == 'compound:prt'):
#                 compoundprt_combined_word = governor_word + '_' + dependent_word
#                 features.append(compoundprt_combined_word)

#     features=list(set(features))
#         # print(features)
#     for item in features:
#         tokens=item.split('_')
#         flag=False
#         for t in tokens:
#             if t in spam:
#                 features.remove(item)
#                 flag=True
#                 break
#         if not flag:
#             feature_matrix.append(item)
#     # print(features)
#     # feature_matrix.append(features)
#     # print(feature_matrix)
#     return all_features, feature_matrix

def ques_feats(path = None):
    # qs, intents, orig=get_qs(False,'/home/lexica/chatbot/intent/training_data/msg.txt','/home/lexica/chatbot/intent/training_data/intent.txt')
    if path and path.endswith('.csv') is False:
        sys.exit("Please use valid .csv extension for training. Void argument for default training data. Breaking")
    questions, intents, orig=get_qs(False,path)
    output={'Text':[],'Extracted Question':[], 'Question Features':[], "Message Features": [], "Intent": []}
    ctr=0
    for q,i in zip(questions,intents):
        ma, mfm = new_feature_engineering_test(orig[ctr])
        output['Message Features'].append(mfm)
        output['Text'].append(orig[ctr])
        output['Intent'].append(i)
        if q.strip():
            print(q)
            all_features, feature_matrix = new_feature_engineering_test(q)
            output['Extracted Question'].append(q)
            output['Question Features'].append(feature_matrix)
        else:
            output['Extracted Question'].append("None")
            output['Question Features'].append("None")
        ctr=ctr+1
    opt=pd.DataFrame(output)
    opt.to_csv('reports/check.csv')
    splice_c = ['Intent', 'Message Features', 'Question Features']
    r_in = opt[splice_c]
    out_file = open('reports/out_results.txt', 'w')
    for tent in r_in[splice_c[0]].unique():
        out_file.write(tent +'\n*******************************\n\n')
        splice_tent = r_in[r_in[splice_c[0]] == tent]
        for col in splice_c[1:]:
            out_file.write(col + '\n-------------------------\n\n')
            comb = [x.strip() for y in splice_tent[col] for x in y]
            count_dict = sorted(dict(Counter(comb)).items(), key = lambda i: i[1], reverse = True)
            for_frame = [{'w' : word, 'f' : freq} for word, freq in dict(count_dict).items()]
            df_to_str = pd.DataFrame(for_frame)[['w', 'f']].sort_values('f', ascending = False).reset_index(drop = True).to_string()
            out_file.write(df_to_str)
            out_file.write('\n\n')        


def testing(path = None): 
    if path and path.endswith('.csv') is False:
        sys.exit("Please use valid .csv extension for training. Void argument for default training data. Breaking")
    questions, intents, bak=get_qs(None, path)
    ctr=0
    for question,lab_cur in zip(questions,intents):
        all_features, feature_matrix = new_feature_engineering_test(question)                
        log_features(all_features, feature_matrix, lab_cur)
        lab_itr=intents[ctr+1]
        ctr=ctr+1
        if lab_cur!=lab_itr:
            global tp
            global fn
            print("%s accuracy:- %s" %(lab_cur,float(tp/(tp+fn))))
            print("*********************")
            print("NEW INTENT - %s" %(lab_itr))
            tp=0
            fn=0
            lab_cur=lab_itr
        
        print('**********************')
        # ctr=ctr+1
        
        print("INTENT - %s" %(lab_cur))

def training(path):
    ext=input("Train using \n [1] Extracted questions \n [2] Original sentence \n")
    print('Starting training \n')
    if ext==1:
        master(True,path)
    else:
        master(False,path)
    print('Training finished \n')
# def user_input():
#     question=input("Enter question\n")
#     question=pre_proc()
#     all_features, feature_matrix = new_feature_engineering(question)
#     log_features(all_features,feature_matrix)


# def pre_proc(sentence):
#     sentence=''.join([i if ord(i) < 128 else ' ' for i in sentence])
#     sentence=sentence.strip()
#     sentence=sentence.replace(' \'',' ')
#     sentence=sentence.replace(' \"',' ')
#     sentence=sentence.replace('\' ',' ')
#     sentence=sentence.replace('\" ',' ')
#     sentence=sentence.replace(',',' , ')
#     sentence=sentence.replace('.',' . ')
#     sentence=sentence.replace('  ',' ')

#     # sentence=sentence.replace(',','')
#     sentence=sentence.lower()
#     phrases=['_excuse_me_','_please_','_help_me_','_can_you_','_hi_','_hello_','_hey_','_good_morning_','_good_evening_','_may_i_','_kindly_','_know_','_can_i_','_should_i_','_i\'m_','_am_','_are_','_was_','_is_','_do_','_did_','_does_','_of_the_','_for_the_','_i_am_','_pls_','_cos_','_i_','_i_want_','_i_need_','_me_','_good_afternoon_','_thank_you_','_thanks_']
#     phrases=sorted(phrases,reverse=True,key=len)
#     temp="_"+sentence.replace(' ','_')+"_"
#     # print(temp)
#     for phrase in phrases:
#         if phrase in temp:
#             temp=temp.replace(phrase,' ')
#     sentence=temp.replace('__','_')
#     sentence=sentence.replace('_',' ')
#     # for word in sentence:
#     #   sing=p.singular_noun(word)
#     #   if sing!=word:
#     #       sentence.replace(word,sing)

#     if(len(sentence)>=3):
#         sentence=str(sentence[1].upper())+sentence[2:].lower()
#     sentence=sentence.strip(' ')
#     # corr=[correction(word) for word in sentence]
#     # sentence=" ".join(corr)
#     # sentence=sentence.strip(string.punctuation())
#     # print(sentence)
#     return sentence





def predict(X,lab):
    cls=open("models/original/intent-logistic-classifier-tree.pickle","rb")
    classifier=pickle.load(cls)
    Y=classifier.predict(X)
    print(Y)
    with open('models/original/intent-index-tree.txt') as file:
        labs=dict(enumerate(file.readlines()))
        print(labs[Y[0]])
        if lab in labs[Y[0]]:
            print("--")
            global tp
            tp = tp +1
        else:
            global fn
            fn= fn +1
op_feats=[]
def log_features(all_features,feature_matrix,lab):
    X = []
    global op_feats
    for q in feature_matrix:
        x = []
        for f in all_features:
            if f in q:
                x.append(1)
                if f not in op_feats:
                    op_feats.append(f)
                print(f)
            else:
                x.append(0)
        X.append(x)
    # print(X)

    # print(X)
    predict(X,lab)
    global tp
    global fn
    # print("acc= %s " %(float(tp/(tp+fn))))


if __name__ == "__main__":
    argvs = sys.argv[1:]
    instruction = 1
    path = None
    if len(argvs):
        if argvs[0] == 'testing' or argvs[0]==2:
            instruction = 2
        elif argvs[0]=='training' or argvs[0]==3:
            instruction=3
        if len(argvs) > 1:
            print ("Thank you for providing a path!")
            path = argvs[1]
    else:
        instruction = input('Please choose option 1, 2 or 3: \n [1] Get feature report \n [2] Model testing \n [3] Model Training \n' )
    if int(instruction) <2:
        ques_feats(path)
    elif int(instruction)==2:
        testing(path)    
    elif int(instruction)==3:
        training(path)

# file_write=open("models/original/opening_hour_features_TEST_new_x.txt","a")
# for feature in op_feats:
#     # print(feature)
#     file_write.write(feature)
#     file_write.write('\n')



