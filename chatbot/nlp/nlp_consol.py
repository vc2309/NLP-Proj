from pycorenlp import StanfordCoreNLP
from question_identifier import language_processor
import re
import nltk
import pickle
# from testing_x import pre_proc
# from extract import get_qs
import numpy as np
import linecache

nlp = StanfordCoreNLP('http://localhost:9000')

def feature_engineering_train(dialogues):
	af = []
	fm = []
	for dialogue in dialogues:
		features = []
		for messages in dialogue:
			orig = []
			lem = []
			pos = []
			origin_tuple = []
			lemma_tuple = []
			for message in messages:
				orig.append(message[0])
				lem.append(message[1])
				pos.append(message[2])
				origin_tuple.append((message[0], message[2]))
				lemma_tuple.append((message[1], message[2]))

			ori_result = parser.parse(origin_tuple)
			for t in ori_result:
				if isinstance(t[0], tuple):
					item = [tt[0].lower() for tt in t]
					for ii in origin_stop_words:
						if ii in item:
							item.remove(ii)
					item = [i for i in item if i.strip()]
					if len(item):	
						features.append('_'.join(item))
						features = features + item
						af.append('_'.join(item))
						af = af + item

			lem_result = parser.parse(lemma_tuple)
			for t in lem_result:
				if isinstance(t[0], tuple):
					item = [tt[0].lower() for tt in t]
					for ii in lemma_stop_words:
						if ii in item:
							item.remove(ii)
					item = [i for i in item if i.strip()]
					if len(item):	
						features.append('_'.join(item))
						features = features + item
						af.append('_'.join(item))
						af = af + item

			af = list(set(af))
		fm.append(features)
	return af, fm

def new_feature_engineering_train(dialogues):
	spam=['i','it','know','what','is','please','pls','ask','tell','me','have']
	all_features = []
	feature_matrix = []
	all_slots = []
	skew_words=['what','library','book','find']
	for line in dialogues:
		print(line)
		sentences = nlp.annotate(line, properties={
			'annotators': 'tokenize,depparse',
			'outputFormat': 'json'}
		).get('sentences')
		# if line=='':
		# 	continue
		features = []
		previous_governor_word=''        
		# smaller sentence is divided by fullstop in each long sentence
		for sentence in sentences:
			splited_line = [i['word'] for i in sentence['tokens']]
			print(' '.join(splited_line)) # the sentence itself                
			for dep_dict in sentence['basicDependencies']:            
				dependent_word = dep_dict['dependentGloss'].lower()
				tag = dep_dict['dep']
				governor_word = dep_dict['governorGloss'].lower()

				if tag in ('nn', 'neg', 'amod', 'acl:relcl', 'nsubjpass'):
					combined_word = dependent_word + '_' + governor_word
					# all_features.append(combined_word)
					features.append(combined_word)
				# TODO: whether 1 word or combined word as feature
				if tag=='advmod':
					
					if dependent_word == "where" or dependent_word=="how":
					
						combined_word = dependent_word + '_' + governor_word
						features.append(combined_word)
						# all_features.append(combined_word)
						# print(combined_word)

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
					# all_features.append(governor_word)
					features.append(nummod_combined)
					features.append(dependent_word)
					features.append(governor_word)
				if(tag == 'nmod'):
					nmod_combined_word = governor_word + '_' + dependent_word
					reverse_nmod_combined_word = dependent_word + '_' + governor_word
					# all_features.append(nmod_combined_word)
					# all_features.append(reverse_nmod_combined_word)  
					features.append(nmod_combined_word)
					# features.append(reverse_nmod_combined_word)
					# e.g. drop off
				if(tag == 'compound'):
					# combining multiple compound
					if(previous_governor_word == governor_word):
						# append the current word first
						current_compound_combined_word = dependent_word + '_' + governor_word
						# all_features.append(current_compound_combined_word)     
						all_slots.append(current_compound_combined_word)
						features.append(current_compound_combined_word)

						all_words = compound_combined_word.split("_")
						all_words.insert((len(all_words)-1), dependent_word)
						compound_combined_word = "_".join(all_words)
						previous_governor_word = governor_word 
					else:
						compound_combined_word = dependent_word + '_' + governor_word                    
						previous_governor_word = governor_word

					# all_features.append(compound_combined_word)
					all_slots.append(compound_combined_word)
					features.append(compound_combined_word)

				if(tag == 'compound:prt'):
					compoundprt_combined_word = governor_word + '_' + dependent_word
					# all_features.append(compoundprt_combined_word)
					features.append(compoundprt_combined_word)
				if(tag == 'compound'):
					all_slots.append(dependent_word)
					all_slots.append(governor_word)
				if(tag == 'amod'):
					amod_combined_names = dependent_word + '_' + governor_word
					all_slots.append(amod_combined_names)
					all_slots.append(governor_word)
			features=list(set(features))
			# print(features)
			for item in features:
				tokens=item.split('_')
				flag=False
				for t in tokens:
					if t in spam:
						tokens.remove(t)
						# features.remove(item)
						flag=True
				if len(tokens):
					i='_'.join(tokens)
					all_features.append(i)
			# print(features)
		# if not len(features):
		# 	feature_matrix.append('')
		# else:
		feature_matrix.append(features)
		# int_dict[intent].append(features)
	all_features = list(set(all_features))
	all_slots = list(set(all_slots))
	return all_features, feature_matrix

def new_feature_engineering_test(question):   
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
                tokens.remove(t)
                flag=True
        if len(tokens):
        	i='_'.join(tokens)
        	feature_matrix.append(i)
    # print(features)
    # feature_matrix.append(features)
    # print(feature_matrix)
    return all_features, feature_matrix

def stanford_tree(line):
	output = nlp.annotate(line, properties={
		'annotators': 'tokenize,ssplit,pos,parse',
		'outputFormat': 'json'
	})
	try:
		return output['sentences']
	except IndexError:
		pass