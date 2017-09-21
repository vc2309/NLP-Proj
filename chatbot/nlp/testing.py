from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pycorenlp import StanfordCoreNLP
#from question_identifier import language_processor
import re
import nltk
import pickle
nlp = StanfordCoreNLP('http://localhost:9000')
tp=0
fn=0
def new_feature_engineering(dialogues,question):
    feature_file=open("models/original/all_features_lower.txt",'r')
    all_features = [i[:len(i)-1] for i in feature_file.readlines()]
    # print(all_features[0:10])
    feature_matrix = []
    # for line in dialogues:
    #     sentences = nlp.annotate(line, properties={
    #         'annotators': 'tokenize,depparse',
    #         'outputFormat': 'json'}
    #     ).get('sentences')

    #     # features = []# smaller sentence is divided by fullstop in each long sentence
    #     for sentence in sentences:
    #         print("yeah")
    #         splited_line = [i['word'] for i in sentence['tokens']]
    #         print(' '.join(splited_line)) # the sentence itself
    #         for dep_dict in sentence['basicDependencies']:
    #             dependent_word = dep_dict['dependentGloss']
    #             tag = dep_dict['dep']
    #             governor_word = dep_dict['governorGloss']
    #             if tag in ('nsubj', 'nn', 'neg', 'nmod', 'amod', 'compound', 'acl:relcl', 'nsubjpass'):
    #                 combined_word = dependent_word + '_' + governor_word
    #                 all_features.append(combined_word)
    #                 # features.append(combined_word)
    #             if(tag == 'dobj'):
    #                 all_features.append(dep_dict['governorGloss'])
    #                 all_features.append(dep_dict['dependentGloss'])
    #                 # features.append(dep_dict['governorGloss'])
    #                 # features.append(dep_dict['dependentGloss'])
    #     #feature_matrix.append(features)
    #     all_features = list(set(all_features))
    features=[]
    qs=nlp.annotate(question, properties={
            'annotators': 'tokenize,depparse',
            'outputFormat': 'json'}
        ).get('sentences')
    for q in qs:
        print("yeah")
        splited_line = [i['word'] for i in q['tokens']]
        print(' '.join(splited_line)) # the q itself
        for dep_dict in q['basicDependencies']:
            dependent_word = dep_dict['dependentGloss']
            tag = dep_dict['dep']
            governor_word = dep_dict['governorGloss']
            if tag in ('nsubj', 'nn', 'neg', 'nmod', 'amod', 'compound', 'acl:relcl', 'nsubjpass'):
                combined_word = dependent_word + '_' + governor_word
                # all_features.append(combined_word)
                features.append(combined_word)
            if(tag == 'dobj'):
                # all_features.append(dep_dict['governorGloss'])
                # all_features.append(dep_dict['dependentGloss'])
                features.append(dep_dict['governorGloss'])
                features.append(dep_dict['dependentGloss'])
    feature_matrix.append(features)
    print(feature_matrix)
    return all_features, feature_matrix

def user_question():
    # question=input("Enter question\n")
    with open('openinghours.txt','r') as file:
        for question in file:
            question=pre_proc(question)
            dialogues=open('training_data/questions.txt', 'r').readlines()
            questions = [s.strip() for s in dialogues if s.strip()]
            all_features, feature_matrix = new_feature_engineering(dialogues,question)
            log_features(all_features,feature_matrix)
            print('**********************')


def pre_proc(sentence):
    sentence=''.join([i if ord(i) < 128 else ' ' for i in sentence])
    sentence=sentence.strip()
    sentence=sentence.replace(' \'',' ')
    sentence=sentence.replace(' \"',' ')
    sentence=sentence.replace('\' ',' ')
    sentence=sentence.replace('\" ',' ')
    # sentence=sentence.replace(',','')
    sentence=sentence.lower()
    phrases=['_excuse_me_','_please_','_help_me_','_can_you_','_hi_','_hello_','_hey_','_good_morning_','_good_evening_','_may_i_','_kindly_','_know_','_can_i_','_should_i_','_i\'m_','_am_','_are_','_was_','_is_','_do_','_did_','_does_','_of_the_','_for_the_']
    temp='_'+sentence.replace(' ','_')+'_'
    for phrase in phrases:
        if phrase in temp:
            temp=temp.replace(phrase,' ')
    sentence=temp.replace('__','_')
    sentence=sentence.replace('_',' ')
    sentence=sentence[0].upper()+sentence[1:].lower()
    sentence=sentence.strip(' ')
    # sentence=sentence.strip(string.punctuation())
    print(sentence)
    return sentence

def predict(X):
    cls=open("models/original/intent-logistic-classifier.pickle","rb")
    classifier=pickle.load(cls)
    Y=classifier.predict(X)
    print(Y)
    with open('models/original/intent-index.txt') as file:
        labs=dict(enumerate(file.readlines()))
        print(labs[Y[0]])
        if "opening hour" in labs[Y[0]]:
            print("--")
            global tp
            tp = tp +1
        else:
            global fn
            fn= fn +1

def log_features(all_features,feature_matrix):
    X = []
    for q in feature_matrix:
        x = []
        for f in all_features:
            if f in q:
                x.append(1)
                print(f)
            else:
                x.append(0)
        X.append(x)
    # print(X)

    # print(X)
    predict(X)
    global tp
    global fn
    print("acc= %s " %(float(tp/(tp+fn))))







user_question()
