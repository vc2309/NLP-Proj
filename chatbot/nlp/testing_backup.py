from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pycorenlp import StanfordCoreNLP
#from question_identifier import language_processor
import re
import nltk
import pickle
import linecache
# from testing_x import pre_proc
from extract import get_qs
import pandas as pd
import numpy as np
nlp = StanfordCoreNLP('http://localhost:9000')
tp=0
fn=0
all_ints={}
def new_feature_engineering(question):
    feature_file=open("models/original/all_features_lower-new_x1.txt",'r')
    all_features = [i[:len(i)-1] for i in feature_file.readlines()]
    feature_matrix = []
    features=[]
    spam=['i','it','know','what','is','please','pls','ask','tell','me','have']
    previous_governor_word=''
    qs=nlp.annotate(question, properties={
            'annotators': 'tokenize,depparse',
            'outputFormat': 'json'}
        ).get('sentences')
    for q in qs:
        print(q['basicDependencies'])
        # print("yeah")
        splited_line = [i['word'] for i in q['tokens']]
        print(' '.join(splited_line)) # the q itself
        for dep_dict in q['basicDependencies']:
            dependent_word = dep_dict['dependentGloss'].lower()
            tag = dep_dict['dep']
            governor_word = dep_dict['governorGloss'].lower()
            if tag in ('nn', 'neg', 'amod', 'acl:relcl', 'nsubjpass'):
                combined_word = dependent_word + '_' + governor_word
                features.append(combined_word)
            if tag=='advmod':
                    if dependent_word == "where" or dependent_word=="how":
                        combined_word = dependent_word + '_' + governor_word
                        features.append(combined_word)
                        # print(combined_word)
            if(tag == 'dobj' or tag=='nsubj'):
                dobj_combined_word = governor_word + '_' + dependent_word
                features.append(dobj_combined_word)
                features.append(governor_word)
                # features.append(dependent_word)
            if tag in ('tmod','nmod:tmod'):
                tmod_combined=governor_word+'_'+dependent_word
                features.append(tmod_combined)
                features.append(dependent_word)

            if tag=='nummod':
                nummod_combined=governor_word+'_'+dependent_word
                features.append(nummod_combined)
                features.append(governor_word)
                features.append(dependent_word)
            if(tag == 'nmod'):
                nmod_combined_word = governor_word + '_' + dependent_word
                reverse_nmod_combined_word = dependent_word + '_' + governor_word
                features.append(nmod_combined_word)


            if(tag == 'compound'):
                    # combining multiple compound
                    if(previous_governor_word == governor_word):
                        # append the current word first
                        current_compound_combined_word = dependent_word + '_' + governor_word
                        features.append(current_compound_combined_word)

                        all_words = compound_combined_word.split("_")
                        all_words.insert((len(all_words)-1), dependent_word)
                        compound_combined_word = "_".join(all_words)
                        previous_governor_word = governor_word 
                    else:
                        compound_combined_word = dependent_word + '_' + governor_word                    
                        previous_governor_word = governor_word
                    features.append(compound_combined_word)
            if(tag == 'compound:prt'):
                compoundprt_combined_word = governor_word + '_' + dependent_word
                features.append(compoundprt_combined_word)

    features=list(set(features))
        # print(features)
    for item in features:
        tokens=item.split('_')
        flag=False
        for t in tokens:
            if t in spam:
                features.remove(item)
                flag=True
                break
        if not flag:
            all_features.append(item)
    # print(features)

    feature_matrix.append(features)
    # print(feature_matrix)
    return all_features, feature_matrix

def ques_feats():
    qpath='/home/lexica/chatbot/intent/training_data/msg.txt'
    ipath='/home/lexica/chatbot/intent/training_data/intents_new.txt'
    questions, intents,orig=get_qs(True,qpath,ipath)
    # print(questions)
    output={'Text':[],'Prepoc Text':[], 'Features':[], 'Intent':[], }
    for i,q in enumerate(questions):
        if q!="":
            print(q)
            all_features, feature_matrix = new_feature_engineering(q)
            output['Prepoc Text'].append(q)
            output['Features'].append(feature_matrix)
            output['Intent'].append('')
            output['Text'].append(orig[i])
    opt=pd.DataFrame(output)
    opt.to_csv('Q_F2f.csv')

def user_question():
    # question=input("Enter question\n")
    # all_features, feature_matrix = new_feature_engineering(question)
    # log_features(all_features,feature_matrix)
    ctr=1
    file2=open('/home/lexica/chatbot/intent/training_data/is.txt','r')
    with open('/home/lexica/chatbot/intent/training_data/qs.txt','r') as file:
        
        lab_cur=linecache.getline('/home/lexica/chatbot/intent/training_data/is.txt',ctr,module_globals=None)
        lab_cur=lab_cur.strip()
        lab_itr=lab_cur
        for question in file:
            ctr=ctr+1
            lab_itr=linecache.getline('/home/lexica/chatbot/intent/training_data/is.txt',ctr,module_globals=None)
            lab_itr=lab_itr.strip()
            question=pre_proc(question)
            if lab_itr=='':
                print("empty %s" %(ctr))
                continue
            if question=='':
                # ctr=ctr+1
                print('OooOoOoooooOoo')
                print(ctr)
                continue
            # dialogues=open('training_data/questions.txt', 'r').readlines()
            # questions = [s.strip() for s in dialogues if s.strip()]
            print('**********************')
            print("INTENT - %s" %(lab_cur))
            all_features, feature_matrix = new_feature_engineering(question)                
            log_features(all_features,feature_matrix,lab_cur)
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
    cls=open("models/original/intent-logistic-classifier-new_x1.pickle","rb")
    classifier=pickle.load(cls)
    Y=classifier.predict(X)
    print(Y)
    with open('models/original/intent-index-new_x1.txt') as file:
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






# user_question()
ques_feats()
# file_write=open("models/original/opening_hour_features_TEST_new_x1.txt","a")
# for feature in op_feats:
#     # print(feature)
#     file_write.write(feature)
#     file_write.write('\n')

