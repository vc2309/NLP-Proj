import re, string
from pycorenlp import StanfordCoreNLP
import json
from nltk.tree import *
from nltk import RegexpParser
import pandas as pd
import numpy as np
from nltk.tag import pos_tag
import simplejson as json

nlp = StanfordCoreNLP('http://192.168.0.100:9000')

def stanford_tree(line):
	output = nlp.annotate(line, properties={
		'annotators': 'tokenize,ssplit,pos,parse',
		'outputFormat': 'json'
	})
	try:
		return output['sentences']
	except IndexError:
		pass

NN_grammar = r"""
	'Noun_phrase' : {<NN.*>+}
	"""

np_parser = RegexpParser(NN_grammar)

def get_np(parse_tree):
	if isinstance(parse_tree, Tree):
		all_np = []
		get_tokens = parse_tree.pos()
		fish_np = np_parser.parse(get_tokens)
		for obj in fish_np:
			if isinstance(obj, Tree):
				np_items = [x[0] for x in obj]
				all_np.append(' '.join(np_items))
		return all_np


# read_unmatched = open('whatsapp_unmatched.txt', 'r').read().split('\n')

# for line in read_unmatched:
# 	if isinstance(line,str):
# 		try:
# 			str_tree = stanford_tree(line)
# 			obj = Tree.fromstring(str_tree)
# 			clause_criteria = ['SBARQ', 'SBAR', 'SQ']
# 			candidates = list(obj.subtrees(filter = lambda x: x.label() in clause_criteria))
# 			if len(candidates) >= 1:
# 				print (candidates[0], candidates[0].label())
# 		except:
# 			pass



# out_file_unmatched = open('test_q_fail.txt', 'w')
# out_file_unmatched = open('wap_fail.txt', 'w')

pd_rows = []
# read_wap = pd.read_csv('whatsapp_data/whatsapp_test.csv', index_col = 0)
# for ix, line in enumerate(read_wap[read_wap.columns[2]]):
# 		if isinstance(line, str):

def extract_subtree(tag, labels, tree, pts):
	ind = 0
	match = []
	st = ''
	if tag in labels:
		ind = labels.index(tag)
		for i, k in enumerate(tree.subtrees()):
			if i == ind:
				subtree_labels = [t.label() for t in k.subtrees()]
				st = ' '.join(subtree_labels)
				go_ahead = True
				if tag == 'SBAR':
					check_labels = [t for t in subtree_labels if t in ['WHADVP', 'WHNP', 'WHPP']]
					if len(check_labels) == 0:
						go_ahead = False
						st = ''
				print(k.leaves())
				print(st)
				no_match = go_ahead
				if len(pts) > 0:
					for pt in pts:
						if st.find(pt) > -1 or pt.find(st) > -1:
							no_match = False
				print(no_match)
				if no_match:
					match = k
				else:
					ind = 0
	st = [st] if st.strip() else []
	return ind, st, match

def append_tree(qs, nps, i, mt):
	if i > 0:
		np = get_np(mt)
		nps = nps + np if np is not None else nps
		qs.append(' '.join(mt.leaves()))
	return qs, nps

unmatch_crit = ['whose', 'that', 'who', 'which']

def unmatch_criteria(m_subtree):
	if any([uc in m_subtree.leaves()[:2] for uc in unmatch_crit]):
		pass
	else:
		return m_subtree

with open('../training_data/testing_questions.txt', 'r') as infile:
	for ix, line in enumerate(infile.read().split('\n')):
		if line:
			print('Get started')
			print(line)
			df_row = {'text' : line}
			line = line.lower()
			# naive_s_tokenize = len(re.split(r'[!\?\.]', line))
			
			turn_I_to_prp = re.sub(r'\bi\b', 'I', line)
			#otherwise is I is labelled 'NN'
			sentences = stanford_tree(turn_I_to_prp)
			df_row['number_sentence_tokens'] = len(sentences)
			questions = []
			noun_phrases = []
			matched_patterns = []
			for sentence in sentences:
				parse_string = sentence['parse']
				p_tree = Tree.fromstring(parse_string)
				print(p_tree)
				labels = [t.label() for t in p_tree.subtrees()]
				print(labels)
				ind, slabels, matched_tree = extract_subtree('SBARQ', labels, p_tree, [])
				if matched_tree != [] and unmatch_criteria(matched_tree):
					matched_patterns.extend(slabels)
					questions, noun_phrases = append_tree(questions, noun_phrases, ind, matched_tree)
				ind, slabels, matched_tree = extract_subtree('SBAR', labels, p_tree, matched_patterns)
				if matched_tree != [] and unmatch_criteria(matched_tree):
					matched_patterns.extend(slabels)
					questions, noun_phrases = append_tree(questions, noun_phrases, ind, matched_tree)
				ind, slabels, matched_tree = extract_subtree('SQ', labels, p_tree, matched_patterns)
				if matched_tree != [] and unmatch_criteria(matched_tree):
					matched_patterns.extend(slabels)
					questions, noun_phrases = append_tree(questions, noun_phrases, ind, matched_tree)
			print(questions)
			df_row['noun_phrase'] = noun_phrases
			# questions contains extracted questions as a list
			df_row['questions'] = ' | '.join(questions)
			pd_rows.append(df_row)
to_frame = pd.DataFrame(pd_rows)
to_frame.to_csv(open('wap_fail.csv', 'w'))