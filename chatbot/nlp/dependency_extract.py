"""
prepossessing 
strip hi and similar greetings, puntuation except full stop and comma, no exscuse me kind of greeting, no please, first word capitalize

TODO:
intent: 
next step add dobj_ in the feature word

slot:

"""

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://192.168.0.100:9000')

with open("sample.txt") as f:
    all_features = []
    feature_matrix = []    
    all_slots = []
    for line in f:
        sentences = nlp.annotate(line, properties={
            'annotators': 'tokenize,depparse',
            'outputFormat': 'json'}
        ).get('sentences')
        
        features = []        
        previous_governor_word =""
        # smaller sentence is divided by fullstop in each long sentence
        for sentence in sentences:          
            splited_line = [i['word'] for i in sentence['tokens']]
            # print(' '.join(splited_line)) # the sentence itself        
            for dep_dict in sentence['basicDependencies']:            
                dependent_word = dep_dict['dependentGloss'].lower()
                tag = dep_dict['dep']
                governor_word = dep_dict['governorGloss'].lower()

                ########################################################################################## logistic
                
                # neg put both to be more specific on the verb that is negative
                if tag in ('nn', 'neg', 'amod', 'acl:relcl', 'nsubjpass'):
                    combined_word = dependent_word + '_' + governor_word
                    all_features.append(combined_word)                    
                    features.append(combined_word)
                
                # TODO: whether 1 word or combined word as feature
                if(tag == 'dobj'):
                    dobj_combined_word = governor_word + '_' + dependent_word
                    all_features.append(dobj_combined_word)
                    all_features.append(governor_word)
                    all_features.append(dependent_word)
                    features.append(dobj_combined_word)
                    features.append(governor_word)
                    features.append(dependent_word)
                                
                if(tag == 'nmod'):
                    nmod_combined_word = governor_word + '_' + dependent_word
                    reverse_nmod_combined_word = dependent_word + '_' + governor_word
                    all_features.append(nmod_combined_word)
                    all_features.append(reverse_nmod_combined_word)  
                    features.append(nmod_combined_word)
                    features.append(reverse_nmod_combined_word)  

                # many examples are weird
                # if(tag == 'acl'):
                #     acl_combined_word = governor_word + '_' + dependent_word
                #     all_features.append(acl_combined_word)

                # i dont think the word I, my sister matters as much to intent => might be relevant to discovering slot
                # if(tag == 'nsubj'):
                #     all_features.append(dependent_word)
                #     features.append(dependent_word)
                          
                ########################################################################################## logistic

                ########################################################################################## logistic + slots
                if(tag == 'compound'):                    
                    # combining multiple compound
                    if(previous_governor_word == governor_word):
                        # append the current word first
                        current_compound_combined_word = dependent_word + '_' + governor_word
                        all_features.append(current_compound_combined_word)     
                        all_slots.append(current_compound_combined_word)
                        features.append(current_compound_combined_word)

                        all_words = compound_combined_word.split("_")
                        all_words.insert((len(all_words)-1), dependent_word)
                        compound_combined_word = "_".join(all_words)
                        previous_governor_word = governor_word 
                    else:
                        compound_combined_word = dependent_word + '_' + governor_word                    
                        previous_governor_word = governor_word                                        
                    
                    all_features.append(compound_combined_word)
                    all_slots.append(compound_combined_word)
                    features.append(compound_combined_word)
                                
                # e.g. drop off
                if(tag == 'compound:prt'):
                    compoundprt_combined_word = governor_word + '_' + dependent_word
                    all_features.append(compoundprt_combined_word)
                    # all_slots.append(compoundprt_combined_word)
                    features.append(compoundprt_combined_word)

                ########################################################################################## logistic + slots

                ########################################################################################## slots
                if(tag == 'compound'):
                    all_slots.append(dependent_word)
                    all_slots.append(governor_word) 
                if(tag == 'amod'):
                    amod_combined_names = dependent_word + '_' + governor_word
                    all_slots.append(amod_combined_names)
                    all_slots.append(governor_word)
                
                # these i guess for real time detection
                # if(tag == 'dobj'):
                #     all_slots.append(dependent_word)
                # if(tag == 'nmod'):
                #     all_slots.append(dependent_word)

                ########################################################################################## slots

        feature_matrix.append(features)    
    all_features = list(set(all_features))
    all_slots = list(set(all_slots)) # all slots is not for intent classification

# all_feature_file = open('log_features.txt', 'w+')
# all_feature_file.write('\n'.join(all_features))
# all_feature_file.close()


# all_slots_file = open('log_slots.txt', 'w+')
# all_slots_file.write('\n'.join(all_slots))
# all_slots_file.close()

# print(all_features)
# print(feature_matrix)

# X = []
# for q in feature_matrix:
#     x = []
#     for f in all_features:
#         if f in q:
#             x.append(1)
#         else:
#             x.append(0)
#     X.append(x)


# print('Logistic Regression')
# logistic_clf = LogisticRegression()
# logistic_clf.fit(X, Y)
# print('RandomForest')
# random_clf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
# random_clf.fit(X, Y)

