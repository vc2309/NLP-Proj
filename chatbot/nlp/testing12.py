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
def new_feature_engineering(question):
    feature_file=open("models/original/all_features_lower-new3.txt",'r')
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
                # all_features.append(combined_word)
                features.append(combined_word)
            if(tag == 'dobj' or tag=='nsubj'):
                dobj_combined_word = governor_word + '_' + dependent_word
                # all_features.append(dobj_combined_word)
                # all_features.append(governor_word)
                # all_features.append(dependent_word)
                features.append(dobj_combined_word)
                features.append(governor_word)
                # features.append(dependent_word)
            if tag in ('tmod','nmod:tmod'):
                tmod_combined=governor_word+'_'+dependent_word
                # all_features.append(tmod_combined)
                # all_features.append(dependent_word)
                features.append(tmod_combined)
                features.append(dependent_word)

            if tag=='nummod':
                nummod_combined=governor_word+'_'+dependent_word
                # all_features.append(nummod_combined)
                # all_features.append(dependent_word)
                features.append(nummod_combined)
                features.append(governor_word)
                features.append(dependent_word)
            if(tag == 'nmod'):
                nmod_combined_word = governor_word + '_' + dependent_word
                reverse_nmod_combined_word = dependent_word + '_' + governor_word
                # all_features.append(nmod_combined_word)
                # all_features.append(reverse_nmod_combined_word)  
                # features.append(nmod_combined_word)
                features.append(reverse_nmod_combined_word)
            if(tag == 'compound'):
                    # combining multiple compound
                    if(previous_governor_word == governor_word):
                        # append the current word first
                        current_compound_combined_word = dependent_word + '_' + governor_word
                        # all_features.append(current_compound_combined_word)     
                        # all_slots.append(current_compound_combined_word)
                        features.append(current_compound_combined_word)

                        all_words = compound_combined_word.split("_")
                        all_words.insert((len(all_words)-1), dependent_word)
                        compound_combined_word = "_".join(all_words)
                        previous_governor_word = governor_word 
                    else:
                        compound_combined_word = dependent_word + '_' + governor_word                    
                        previous_governor_word = governor_word
                    # all_features.append(compound_combined_word)
                    # all_slots.append(compound_combined_word)
                    features.append(compound_combined_word)
            if(tag == 'compound:prt'):
                compoundprt_combined_word = governor_word + '_' + dependent_word
                # all_features.append(compoundprt_combined_word)
                features.append(compoundprt_combined_word)

    feature_matrix.append(features)
    print(feature_matrix)
    return all_features, feature_matrix

def user_question():
    # question=input("Enter question\n")
    # all_features, feature_matrix = new_feature_engineering(question)
    # log_features(all_features,feature_matrix)
    ctr=0
    with open('training_data/questions_new.txt','r') as file:
        for question in file:
            if(ctr>=1037 and ctr<1057):
                question=pre_proc(question)
                # dialogues=open('training_data/questions.txt', 'r').readlines()
                # questions = [s.strip() for s in dialogues if s.strip()]
                all_features, feature_matrix = new_feature_engineering(question)
                log_features(all_features,feature_matrix)
                print('**********************')
            ctr=ctr+1


def pre_proc(sentence):
    sentence=''.join([i if ord(i) < 128 else ' ' for i in sentence])
    sentence=sentence.strip()
    sentence=sentence.replace(' \'',' ')
    sentence=sentence.replace(' \"',' ')
    sentence=sentence.replace('\' ',' ')
    sentence=sentence.replace('\" ',' ')
    sentence=sentence.replace(',',' , ')
    sentence=sentence.replace('.',' . ')
    sentence=sentence.replace('  ',' ')

    # sentence=sentence.replace(',','')
    sentence=sentence.lower()
    phrases=['_excuse_me_','_please_','_help_me_','_can_you_','_hi_','_hello_','_hey_','_good_morning_','_good_evening_','_may_i_','_kindly_','_know_','_can_i_','_should_i_','_i\'m_','_am_','_are_','_was_','_is_','_do_','_did_','_does_','_of_the_','_for_the_','_i_am_','_pls_','_cos_','_i_','_i_want_','_i_need_','_me_','_good_afternoon_','_thank_you_','_thanks_']
    phrases=sorted(phrases,reverse=True,key=len)
    temp="_"+sentence.replace(' ','_')+"_"
    # print(temp)
    for phrase in phrases:
        if phrase in temp:
            temp=temp.replace(phrase,' ')
    sentence=temp.replace('__','_')
    sentence=sentence.replace('_',' ')
    # for word in sentence:
    #   sing=p.singular_noun(word)
    #   if sing!=word:
    #       sentence.replace(word,sing)

    if(len(sentence)>=3):
        sentence=str(sentence[1].upper())+sentence[2:].lower()
    sentence=sentence.strip(' ')
    # corr=[correction(word) for word in sentence]
    # sentence=" ".join(corr)
    # sentence=sentence.strip(string.punctuation())
    # print(sentence)
    return sentence

def predict(X):
    cls=open("models/original/intent-logistic-classifier-new3.pickle","rb")
    classifier=pickle.load(cls)
    Y=classifier.predict(X)
    print(Y)
    with open('models/original/intent-index-new3.txt') as file:
        labs=dict(enumerate(file.readlines()))
        print(labs[Y[0]])
        if "searching for a missing item" in labs[Y[0]]:
            print("--")
            global tp
            tp = tp +1
        else:
            global fn
            fn= fn +1
op_feats=[]
def log_features(all_features,feature_matrix):
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
    predict(X)
    global tp
    global fn
    print("acc= %s " %(float(tp/(tp+fn))))






user_question()
file_write=open("models/original/opening_hour_features_TEST_new3.txt","a")
for feature in op_feats:
    print(feature)
    file_write.write(feature)
    file_write.write('\n')

