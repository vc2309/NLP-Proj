#!/usr/bin/python
# -*- coding: utf-8 -*-
# from sklearn import tree
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

from chatbot.nlp.question_identifier import language_processor
# import re
import nltk
import pickle

from chatbot.helper import debug

questions = []
intents = []

with open('resources/sorted_unique_slot.pkl', 'rb') as pkl:
	sortedlist = pickle.load(pkl)


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
			# 	debug(sentence)
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
					item = [i for i in item if i.strip().rstrip() != '']
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
					item = [i for i in item if i.strip().rstrip() != '']
					if len(item):
						features.append('_'.join(item))
						features = features + item
						af.append('_'.join(item))
						af = af + item

			af = list(set(af))
		fm.append(features)
	return af, fm

# dialogues, mm = nlp_preprocess(questions)
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

# debug(len(feature_matrix))
# debug(len(intents))

# debug('Logistic Regression')
# logistic_clf = LogisticRegression()
# logistic_clf.fit(X, Y)
# debug('RandomForest')
# random_clf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
# random_clf.fit(X, Y)


# f = open("intents/original/intent-logistic-classifier.pickle", "wb")
# pickle.dump(logistic_clf , f)

# ff = open("intents/original/intent-random-classifier.pickle", "wb")
# pickle.dump(random_clf, ff)

# intent_index_file = open("intents/original/intent-index.txt", 'w+')
# intent_index_file.write('\n'.join(unique_intents))
# intent_index_file.close()

# all_feature_file = open('intents/original/all_features.txt', 'w+')
# all_feature_file.write('\n'.join(all_features))
# all_feature_file.close()

# ORIGINAL WORD
with open(
	'resources/intents/original/intent-logistic-classifier.pickle',
	'rb'
) as pkl:
	original_logistic_clf = pickle.load(pkl)

with open(
	'resources/intents/original/intent-random-classifier.pickle',
	'rb'
) as pkl:
	original_random_clf = pickle.load(pkl)

with open(
	'resources/intents/original/intent-index.txt', 'r'
) as txt:
	original_unique_intents = [oui.strip() for oui in txt.readlines()]

with open(
	'resources/intents/original/all_features.txt', 'r'
) as txt:
	original_all_features = [oaf.strip() for oaf in txt.readlines()]
# ORIGINAL WORD END

# LABEL WORD
with open(
	'resources/intents/labels/intent-logistic-classifier.pickle',
	'rb'
) as pkl:
	label_logistic_clf = pickle.load(pkl)

with open(
	'resources/intents/labels/intent-random-classifier.pickle',
	'rb'
) as pkl:
	label_random_clf = pickle.load(pkl)

with open(
	'resources/intents/labels/intent-index.txt', 'r'
) as txt:
	label_unique_intents = [lui.strip() for lui in txt.readlines()]

with open(
	'resources/intents/labels/all_features.txt', 'r'
) as txt:
	label_all_features = [laf.strip() for laf in txt.readlines()]
# LABEL WORD END

# COMBINE WORD
# f = open('intents/combined/intent-logistic-classifier.pickle', 'rb')
# combined_logistic_clf = pickle.load(f)
# f.close()
# f = open('intents/combined/intent-random-classifier.pickle', 'rb')
# combined_random_clf = pickle.load(f)
# f.close()
# combined_unique_intents = open('intents/combined/intent-index.txt', 'r').readlines()
# combined_unique_intents = [u.strip().rstrip() for u in combined_unique_intents]
# combined_all_features = open('intents/combined/all_features.txt', 'r').readlines()
# combined_all_features = [a.strip().rstrip() for a in combined_all_features]
# COMBINE WORD END

def original_predict(dialogue_tuple):
	debug("Original predict")
	target_features, target_feature_matrix = feature_engineering(dialogue_tuple)
	target_x = []
	for tf in target_feature_matrix:
		tx = []
		for f in original_all_features:
			if f in tf:
				tx.append(1)
			else:
				tx.append(0)
		target_x.append(tx)
	random_result = original_random_clf.predict_proba(target_x)
	logistic_result = original_logistic_clf.predict_proba(target_x)
	resp_dict = {}
	resp_list = []
	for ri in range(len(logistic_result[0])):
		resp_dict[original_unique_intents[ri]] = (random_result[0][ri], logistic_result[0][ri])
		resp_list.append({"intent": original_unique_intents[ri], "random": random_result[0][ri], "logistic": logistic_result[0][ri], "score": (random_result[0][ri] + logistic_result[0][ri]) / 2})
	sorted_resp_list = sorted(resp_list, key=lambda k: k["score"])
	sorted_resp_list.reverse()
	debug(sorted_resp_list)
	for ri in range(len(sorted_resp_list)):
		sorted_resp_list[ri]['random'] = str(sorted_resp_list[ri]['random'])
		sorted_resp_list[ri]['logistic'] = str(sorted_resp_list[ri]['logistic'])
		sorted_resp_list[ri]['score'] = str(sorted_resp_list[ri]['score'])
	return sorted_resp_list

def labels_predict(sentence_tokens, dialogue_tuple):
	debug("Label predict")
	actual_tokens = []
	for sent in sentence_tokens:
		for token in sent['tokens']:
			if token['lemma'].lower() == 'book' and token['pos'].startswith('V'):
				actual_tokens.append('|'.join(list(token['originalText'].lower())))
			else:
				actual_tokens.append(token['originalText'].lower())

	testing_question = ' ' + ' '.join(actual_tokens) + ' '
	for item in sortedlist:
		if testing_question.find(' '+item[0]+' ') > -1:
			testing_question = testing_question.replace(' '+item[0]+' ', ' '+item[1]+' ')
	testing_question = testing_question.replace('|', '')
	testing_question_tokens = testing_question.split()
	gc = 0
	new_list_for_sentences = []
	for ds in range(len(dialogue_tuple[0])):
		new_list_for_sentence = []
		for ts in range(len(dialogue_tuple[0][ds])):
			pos = dialogue_tuple[0][ds][ts][0].split('_')[0]
			word = dialogue_tuple[0][ds][ts][0].split('_')[1]
			new_list_for_sentence.append((pos + '_' + testing_question_tokens[gc], dialogue_tuple[0][ds][ts][1], dialogue_tuple[0][ds][ts][2]))
			gc += 1
		new_list_for_sentences.append(new_list_for_sentence)

	debug(new_list_for_sentences)
	target_features, target_feature_matrix = feature_engineering([new_list_for_sentences])
	target_x = []
	for tf in target_feature_matrix:
		tx = []
		for f in label_all_features:
			if f in tf:
				tx.append(1)
			else:
				tx.append(0)
		target_x.append(tx)
	random_result = label_random_clf.predict_proba(target_x)
	logistic_result = label_logistic_clf.predict_proba(target_x)
	resp_dict = {}
	resp_list = []
	for ri in range(len(logistic_result[0])):
		resp_dict[label_unique_intents[ri]] = (random_result[0][ri], logistic_result[0][ri])
		resp_list.append({"intent": label_unique_intents[ri], "random": random_result[0][ri], "logistic": logistic_result[0][ri], "score": (random_result[0][ri] + logistic_result[0][ri]) / 2})
	sorted_resp_list = sorted(resp_list, key=lambda k: k["score"])
	sorted_resp_list.reverse()
	debug(sorted_resp_list)
	for ri in range(len(sorted_resp_list)):
		sorted_resp_list[ri]['random'] = str(sorted_resp_list[ri]['random'])
		sorted_resp_list[ri]['logistic'] = str(sorted_resp_list[ri]['logistic'])
		sorted_resp_list[ri]['score'] = str(sorted_resp_list[ri]['score'])
	return sorted_resp_list

def predict(target_message):
	# debug(target_message)
	pre, stanford_responses = nlp_preprocess([target_message])
	stanford_response = stanford_responses[0]

	label_result = labels_predict(stanford_response, pre)
	original_result = original_predict(pre)
	debug(label_result[:3])
	debug(original_result[:3])
	return label_result, original_result, stanford_responses


# predict('I am looking for visting the law library this coming thursday and next tuesday. Can I')