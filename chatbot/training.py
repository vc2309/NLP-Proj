# -*- coding: utf-8 -*-
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pycorenlp import StanfordCoreNLP
from question_identifier import language_processor
import re
import nltk
import pickle


# nlp = StanfordCoreNLP('http://192.168.0.100:9000')
nlp = StanfordCoreNLP('http://localhost:9000')

unique_slots = open('training_data/unique_slot1.txt', 'r').read()
unique_slots = unique_slots.split('LEXICA_')
unique_slots = [q for q in unique_slots if q.strip()]

slots_dict = {}
for u in unique_slots:
	u = u.split('\n')
	u = [uu for uu in u if uu.strip()]
	key = u[0]
	values = u[1:]
	for v in values:
		v_count = v.count(' ')
		key_write = ''+key
		for vc in range(v_count):
			key_write = key_write + ' ' + key
		slots_dict[v] = key_write.strip()

for key in slots_dict.keys():
	print(key)
	print(slots_dict[key])

newlist = slots_dict.items()
sortedlist = sorted(newlist, key=lambda s: len(s[0]))
sortedlist.reverse()

with open('sorted_unique_slot.pkl', 'wb') as pkl:
	pickle.dump(sortedlist , pkl)

def nlp_preprocess(qs):
	d = []
	messages_with_labels = language_processor(qs)
	for q in messages_with_labels:
		dialogue = []
		for sent in q:
			sentence = []
			sentence_pos = []
			for token in sent['tokens']:
				sentence.append((token['pos'][0] + '_' + token['originalText'], token['pos'][0] + '_' + token['lemma'], token['pos']))
				sentence_pos.append(token['pos'])
			pos_sent = ',' + ','.join(sentence_pos)
			dialogue.append(sentence)
			# if (pos_sent.find(',V') > -1 and pos_sent.find(',N') > -1) == True:
			# 	dialogue.append(sentence)
			# else:
			# 	print(sentence)
		d.append(dialogue)
	return d, messages_with_labels

grammer = r"""
	QL: { <WRB><JJ> }
	Q: { <WRB|WP> }
	NN: { <NN.*>+ }
	VB: { <VB.*> }
	JJ: { <JJ.* >}
	"""
parser = nltk.RegexpParser(grammer)

origin_stop_words = ['v_please', 'n_hi', 'v_is', 'v_are', 'v_am', 'v_was', 'v_were', 'v_do', 'v_did', 'v_does', "v_'s", "v_'d", "v_'m", 'v_let', 'v_know']
lemma_stop_words = ['v_please', 'n_hi', 'v_be', 'v_do', 'v_let', 'v_know']

def feature_engineering(dialogues):
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

def new_feature_engineering(dialogues):
	all_features = []
	feature_matrix = []
	for line in dialogues:
		sentences = nlp.annotate(line, properties={
			'annotators': 'tokenize,depparse',
			'outputFormat': 'json'}
		).get('sentences')
		
		features = []        
		# smaller sentence is divided by fullstop in each long sentence
		for sentence in sentences:
			print("yeah")
			splited_line = [i['word'] for i in sentence['tokens']]
			print(' '.join(splited_line)) # the sentence itself                
			for dep_dict in sentence['basicDependencies']:            
				dependent_word = dep_dict['dependentGloss']
				tag = dep_dict['dep']
				governor_word = dep_dict['governorGloss']

				if tag in ('nsubj', 'nn', 'neg', 'nmod', 'amod', 'compound', 'acl:relcl', 'nsubjpass'):
					combined_word = dependent_word + '_' + governor_word
					all_features.append(combined_word)
					features.append(combined_word)
				if(tag == 'dobj'):
					all_features.append(dep_dict['governorGloss'])
					all_features.append(dep_dict['dependentGloss'])
					features.append(dep_dict['governorGloss'])
					features.append(dep_dict['dependentGloss'])            
		feature_matrix.append(features)
		all_features = list(set(all_features))
	return all_features, feature_matrix

questions = []
intents = []

questions = open('training_data/questions.txt', 'r').readlines()
intents = open('training_data/intents.txt', 'r').readlines()
questions = [s.strip() for s in questions if s.strip()]
intents = [s.strip() for s in intents if s.strip()]

unique_intents = list(set(intents))
print(unique_intents)
Y = []
for i in intents:
	Y.append(unique_intents.index(i))

# dialogues, mm = nlp_preprocess(questions)
# all_features, feature_matrix = feature_engineering(dialogues)
all_features, feature_matrix = new_feature_engineering(questions)
X = []
for q in feature_matrix:
	x = []
	for f in all_features:
		if f in q:
			x.append(1)
		else:
			x.append(0)
	X.append(x)

print('Logistic Regression')
logistic_clf = LogisticRegression()
logistic_clf.fit(X, Y)
print('RandomForest')
random_clf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
random_clf.fit(X, Y)


f = open("models/original/intent-logistic-classifier.pickle", "wb")
pickle.dump(logistic_clf , f)

ff = open("models/original/intent-random-classifier.pickle", "wb")
pickle.dump(random_clf, ff)

intent_index_file = open("models/original/intent-index.txt", 'w+')
intent_index_file.write('\n'.join(unique_intents))
intent_index_file.close()

all_feature_file = open('models/original/all_features.txt', 'w+')
all_feature_file.write('\n'.join(all_features))
all_feature_file.close()



# s_questions = open('training_data/questions_with_slot.txt', 'r').readlines()
# s_intents = open('training_data/intents_with_slot.txt', 'r').readlines()
# s_questions = [s.strip() for s in s_questions if s.strip()]
# s_intents = [s.strip() for s in s_intents if s.strip()]

# unique_intents = list(set(s_intents))
# print(unique_intents)
# Y = []
# for i in s_intents:
# 	Y.append(unique_intents.index(i))

# dialogues, mm = nlp_preprocess(s_questions)
# all_features, feature_matrix = feature_engineering(dialogues)

# X = []
# for q in feature_matrix:
# 	x = []
# 	for f in all_features:
# 		if f in q:
# 			x.append(1)
# 		else:
# 			x.append(0)
# 	X.append(x)

# print('Slot Question Logistic Regression')
# logistic_clf = LogisticRegression()
# logistic_clf.fit(X, Y)
# print('Slot Question RandomForest')
# random_clf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
# random_clf.fit(X, Y)


# f = open("intents/labels/intent-logistic-classifier.pickle", "wb")
# pickle.dump(logistic_clf , f)

# ff = open("intents/labels/intent-random-classifier.pickle", "wb")
# pickle.dump(random_clf, ff)

# intent_index_file = open("intents/labels/intent-index.txt", 'w+')
# intent_index_file.write('\n'.join(unique_intents))
# intent_index_file.close()

# all_feature_file = open('intents/labels/all_features.txt', 'w+')
# all_feature_file.write('\n'.join(all_features))
# all_feature_file.close()


# def original_predict(dialogue_tuple):
# 	print("Original predict")
# 	target_features, target_feature_matrix = feature_engineering(dialogue_tuple)
# 	target_x = []
# 	for tf in target_feature_matrix:
# 		tx = []
# 		for f in original_all_features:
# 			if f in tf:
# 				tx.append(1)
# 			else:
# 				tx.append(0)
# 		target_x.append(tx)
# 	random_result = original_random_clf.predict_proba(target_x)
# 	logistic_result = original_logistic_clf.predict_proba(target_x)
# 	resp_dict = {}
# 	resp_list = []
# 	for ri in range(len(logistic_result[0])):
# 		resp_dict[original_unique_intents[ri]] = (random_result[0][ri], logistic_result[0][ri])
# 		resp_list.append({"intent": original_unique_intents[ri], "random": random_result[0][ri], "logistic": logistic_result[0][ri], "score": (random_result[0][ri] + logistic_result[0][ri]) / 2})
# 	sorted_resp_list = sorted(resp_list, key=lambda k: k["score"])
# 	sorted_resp_list.reverse()
# 	print(sorted_resp_list)
# 	for ri in range(len(sorted_resp_list)):
# 		sorted_resp_list[ri]['random'] = str(sorted_resp_list[ri]['random'])
# 		sorted_resp_list[ri]['logistic'] = str(sorted_resp_list[ri]['logistic'])
# 		sorted_resp_list[ri]['score'] = str(sorted_resp_list[ri]['score'])
# 	return sorted_resp_list

# def labels_predict(sentence_tokens, dialogue_tuple):
# 	print("Label predict")
# 	actual_tokens = []
# 	for sent in sentence_tokens:
# 		for token in sent['tokens']:
# 			if token['lemma'].lower() == 'book' and token['pos'].startswith('V'):
# 				actual_tokens.append('|'.join(list(token['originalText'].lower())))
# 			else:
# 				actual_tokens.append(token['originalText'].lower())

# 	testing_question = ' ' + ' '.join(actual_tokens) + ' '
# 	for item in sortedlist:
# 		if testing_question.find(' '+item[0]+' ') > -1:
# 			testing_question = testing_question.replace(' '+item[0]+' ', ' '+item[1]+' ')
# 	testing_question = testing_question.replace('|', '')
# 	testing_question_tokens = testing_question.split()
# 	gc = 0
# 	new_list_for_sentences = []
# 	for ds in range(len(dialogue_tuple[0])):
# 		new_list_for_sentence = []
# 		for ts in range(len(dialogue_tuple[0][ds])):
# 			pos = dialogue_tuple[0][ds][ts][0].split('_')[0]
# 			word = dialogue_tuple[0][ds][ts][0].split('_')[1]
# 			new_list_for_sentence.append((pos + '_' + testing_question_tokens[gc], dialogue_tuple[0][ds][ts][1], dialogue_tuple[0][ds][ts][2]))
# 			gc += 1
# 		new_list_for_sentences.append(new_list_for_sentence)

# 	print(new_list_for_sentences)
# 	target_features, target_feature_matrix = feature_engineering([new_list_for_sentences])
# 	target_x = []
# 	for tf in target_feature_matrix:
# 		tx = []
# 		for f in label_all_features:
# 			if f in tf:
# 				tx.append(1)
# 			else:
# 				tx.append(0)
# 		target_x.append(tx)
# 	random_result = label_random_clf.predict_proba(target_x)
# 	logistic_result = label_logistic_clf.predict_proba(target_x)
# 	resp_dict = {}
# 	resp_list = []
# 	for ri in range(len(logistic_result[0])):
# 		resp_dict[label_unique_intents[ri]] = (random_result[0][ri], logistic_result[0][ri])
# 		resp_list.append({"intent": label_unique_intents[ri], "random": random_result[0][ri], "logistic": logistic_result[0][ri], "score": (random_result[0][ri] + logistic_result[0][ri]) / 2})
# 	sorted_resp_list = sorted(resp_list, key=lambda k: k["score"])
# 	sorted_resp_list.reverse()
# 	print(sorted_resp_list)
# 	for ri in range(len(sorted_resp_list)):
# 		sorted_resp_list[ri]['random'] = str(sorted_resp_list[ri]['random'])
# 		sorted_resp_list[ri]['logistic'] = str(sorted_resp_list[ri]['logistic'])
# 		sorted_resp_list[ri]['score'] = str(sorted_resp_list[ri]['score'])
# 	return sorted_resp_list

# def predict(target_message):
# 	# print(target_message)
# 	pre, stanford_responses = nlp_preprocess([target_message])
# 	stanford_response = stanford_responses[0]

# 	label_result = labels_predict(stanford_response, pre)
# 	original_result = original_predict(pre)
# 	print(label_result[:3])
# 	print(original_result[:3])
# 	return label_result, original_result, stanford_responses
	

# predict('I am looking for visting the law library this coming thursday and next tuesday. Can I')