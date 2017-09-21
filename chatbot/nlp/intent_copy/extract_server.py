import re, string
from pycorenlp import StanfordCoreNLP
import json
from nltk.tree import *
from nltk import RegexpParser
import pandas as pd
import numpy as np
from nltk.tag import pos_tag
import simplejson as json
from nlp_consol import stanford_tree
# def stanford_tree(line):
	# output = nlp.annotate(line, properties={
	# 	'annotators': 'tokenize,ssplit,pos,parse',
	# 	'outputFormat': 'json'
	# })
	# try:
	# 	return output['sentences']
	# except IndexError:
	# 	pass

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
	indices = []
	matches = []
	sts = []
	if tag in labels:
		indices = [i for i, x in enumerate(labels) if x == tag]
		for i, k in enumerate(tree.subtrees()):

			if i in indices:
				subtree_labels = [t.label() for t in k.subtrees()]
				st = ' '.join(subtree_labels)
				go_ahead = True
				if tag == 'SBAR':
					check_labels = [t for t in subtree_labels if t in ['WHADVP', 'WHNP', 'WHPP']]
					if len(check_labels) == 0:
						go_ahead = False
						st = ''
				no_match = go_ahead
				if len(pts) > 0:
					for pt in pts:
						if st.find(pt) > -1 or pt.find(st) > -1:
							no_match = False
				if no_match:
					sts.append(st)
					matches.append(k)
				else:
					indices.remove(i)
	return len(indices), sts, matches

def append_tree(qs, nps, mts):
	for mt in mts:
		np = get_np(mt)
		nps = nps + np if np is not None else nps
		qs.append(' '.join(mt.leaves()))
	return qs, nps

unmatch_crit = ['whose', 'that', 'who', 'which']

def unmatch_criteria(m_subtree):
	unmatch = False
	for m_s in m_subtree:
		if any([uc in m_s.leaves()[:2] for uc in unmatch_crit]):
			pass
		else:
			unmatch = True
	return unmatch

def extract(lines):
	all_qs=[]
	orig_text=[]
	for ix, line in enumerate(lines):
		if line:
			# print('Get started')
			# print(line)
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
				# print(p_tree)
				labels = [t.label() for t in p_tree.subtrees()]
				# print(labels)
				ind, slabels, matched_tree = extract_subtree('SBARQ', labels, p_tree, [])
				if ind and unmatch_criteria(matched_tree):
					matched_patterns.extend(slabels)
					questions, noun_phrases = append_tree(questions, noun_phrases, matched_tree)
				ind, slabels, matched_tree = extract_subtree('SBAR', labels, p_tree, matched_patterns)
				if ind and unmatch_criteria(matched_tree):
					matched_patterns.extend(slabels)
					questions, noun_phrases = append_tree(questions, noun_phrases, matched_tree)
				ind, slabels, matched_tree = extract_subtree('SQ', labels, p_tree, matched_patterns)
				if ind and unmatch_criteria(matched_tree):
					matched_patterns.extend(slabels)
					questions, noun_phrases = append_tree(questions, noun_phrases, matched_tree)
			if not len(questions):
				print(line)
				print('**')
				all_qs.append(line)
			else:
				all_qs.append(' | '.join(questions))
			orig_text.append(line)
			df_row['noun_phrase'] = noun_phrases
			# questions contains extracted questions as a list
			df_row['questions'] = ' | '.join(questions)
			pd_rows.append(df_row)
	to_frame = pd.DataFrame(pd_rows)
	to_frame.to_csv(open('wap_fail.csv', 'w'))
	# print(all_qs)
	return all_qs,orig_text

def pre_proc(sentence):
	sentence=''.join([i if ord(i) < 128 else ' ' for i in sentence])
	sentence=sentence.strip()
	sentence=sentence.replace(' \'',' ')
	sentence=sentence.replace(' \"',' ')
	sentence=sentence.replace('\' ',' ')
	sentence=sentence.replace('\"',' ')
	sentence=sentence.replace(',',' , ')
	sentence=sentence.replace('.',' . ')
	sentence=sentence.replace('  ',' ')
	sentence=sentence.replace('\n',' ').replace('\r','')

	# sentence=sentence.replace(',','')
	sentence=sentence.lower()
	# phrases=['_ok_','_okay_','_excuse__me_','_please_','_help__me_','_can__you_','_hi_','_hello_','_hey_','_good__morning_','_good__evening_','_may__i_','_kindly_','_know_','_can__i_','_should__i_','_i\'m_','_am_','_are_','_was_','_is_','_do_','_did_','_does_','_of__the_','_for__the_','_i__am_','_pls_','_cos_','_i_','_i__want_','_i__need_','_me_','_good__afternoon_','_thank__you_','_thanks_']
	# phrases=sorted(phrases,reverse=True,key=len)
	# temp="_"+sentence.replace(' ','__')+"_"
	# for phrase in phrases:
	#     if phrase in temp:
	#         temp=temp.replace(phrase,' ')
	# sentence=temp.replace('__','_')
	# sentence=sentence.replace('_',' ')
	# sentence=sentence.replace('  ',' ')
	# for word in sentence:
	#   sing=p.singular_noun(word)
	#   if sing!=word:
	#       sentence.replace(word,sing)

	if(len(sentence)>=3):
		sentence=str(sentence[0].upper())+sentence[1:].lower()
	sentence=sentence.strip(' ')
	# print(sentence)
	# corr=[correction(word) for word in sentence]
	# sentence=" ".join(corr)
	# sentence=sentence.strip(string.punctuation())
	# print(sentence)
	# print (sentence)
	return sentence
def get_csv():
	pattern='["A-Za-z]'
	wq=open("/home/lexica/chatbot/intent/training_data/qs.txt","w")
	wi=open("/home/lexica/chatbot/intent/training_data/is.txt","w")
	m=re.compile(pattern)
	questions=[]
	intents=[]
	with open('/home/lexica/chatbot/intent/training_data/questions_new.txt','r') as file:
		# orig=file.readlines().split('\n')
		file2=open('/home/lexica/chatbot/intent/training_data/intents_new.txt','r')
		for question,intent in zip(file,file2):
			# print(question)
			if not m.match(question):
				continue
			if question[0]=='\"':
				q=""
				question=question+" "+q
				# intent=next(file2)
				while '\"' not in q:
					question=question+" "+q
					q=next(file)
					intent=next(file2)
			# print(question)
			#print("INTENT = %s" %(intent))
			question=question.replace('\n', ' ').replace('\r', '')
			wq.write(question)
			wq.write('\n')
			wi.write(intent)
			questions.append(question)
			intents.append(intent)
	questions = [s.strip() for s in questions if s.strip()]
	intents = [s.strip() for s in intents if s.strip()]
	dict_pd={'Content of Messages':questions, 'Technical Intent':intents}
	df=pd.DataFrame(dict_pd)
	df.to_csv('training_data/csv/big_dataset.csv')
	return 'training_data/csv/big_dataset.csv'
def get_qs(opt,csv_path=None):
	if csv_path == None:
		csv_path = 'training_data/csv/filter_use.csv'
	else:
		csv_path = get_csv()
	pattern='["A-Za-z]'
	wq=open("/home/lexica/chatbot/intent/training_data/txt/qs.txt","w")
	wi=open("/home/lexica/chatbot/intent/training_data/txt/is.txt","w")
	m=re.compile(pattern)
	questions=[]
	intents=[]
	three_cols = ['Content of Messages', 'Technical Intent', 'Not FAQ']
	read_in_csv = pd.read_csv(csv_path, index_col=0)
	col1 = read_in_csv[three_cols[0]]
	col2 =  read_in_csv[three_cols[1]]
	for question,intent in zip(col1,col2):
		# print(question)
		if not m.match(question):
			continue
		# if question[0]=='\"':
		# 	q=""
		# 	question=question+" "+q
		# 	# intent=next(file2)
		# 	while '\"' not in q:
		# 		question=question+" "+q
		# 		q=next(file)
		# 		intent=next(file2)
		# print(question)
		#print("INTENT = %s" %(intent))
		question=question.replace('\n', ' ').replace('\r', '')
		wq.write(question)
		wq.write('\n')
		wi.write(intent)
		questions.append(question)
		intents.append(intent)
	questions = [s.strip() for s in questions if s.strip()]
	intents = [s.strip() for s in intents if s.strip()]
	bak,orig=extract(questions)
	if opt:
		questions=[pre_proc_orig(q) for q in questions]
		return questions,intents,orig
	else:
		questions=[pre_proc(q) for q in questions]
		questions,orig=extract(questions)
		filtered_questions= []
		filtered_intents=[]
		filtered_orig=[]
		for i, q in enumerate(questions):
			lines=q.split('|')
			for seg in lines:
				if seg.strip() != orig[i]:
					filtered_questions.append(seg)
					filtered_intents.append(intents[i])
				else:
					filtered_questions.append(orig[i])
					filtered_intents.append(intents[i])
				filtered_orig.append(orig[i])
		print(len(filtered_questions),len(filtered_intents))
		return filtered_questions,filtered_intents,filtered_orig

# print(list(zip(get_qs())))
def pre_proc_orig(sentence):
	sentence=''.join([i if ord(i) < 128 else ' ' for i in sentence])
	sentence=sentence.strip()
	sentence=sentence.replace(' \'',' ')
	sentence=sentence.replace(' \"',' ')
	sentence=sentence.replace('\' ',' ')
	sentence=sentence.replace('\"',' ')
	sentence=sentence.replace(',',' , ')
	sentence=sentence.replace('.',' . ')
	sentence=sentence.replace('  ',' ')
	sentence=sentence.replace('\n',' ').replace('\r','')
	sentence=sentence.lower()
	phrases=['_ok_','_okay_','_excuse__me_','_please_','_help__me_','_can__you_','_hi_','_hello_','_hey_','_good__morning_','_good__evening_','_may__i_','_kindly_','_know_','_can__i_','_should__i_','_i\'m_','_am_','_are_','_was_','_is_','_do_','_did_','_does_','_of__the_','_for__the_','_i__am_','_pls_','_cos_','_i_','_i__want_','_i__need_','_me_','_good__afternoon_','_thank__you_','_thanks_']
	phrases=sorted(phrases,reverse=True,key=len)
	temp="_"+sentence.replace(' ','__')+"_"
   
   
	for phrase in phrases:
		if phrase in temp:
			temp=temp.replace(phrase,' ')
	sentence=temp.replace('__','_')
	sentence=sentence.replace('_',' ')
	sentence=sentence.replace('  ',' ')

	if(len(sentence)>=3):
		sentence=str(sentence[1].upper())+sentence[2:].lower()
	sentence=sentence.strip(' ')
	return sentence



# get_qs()






