#!/usr/bin/python
# -*- coding: utf-8 -*-
import requests
import re

questions = open('questions1.txt', 'r').readlines()
intents = open("intents1.txt", 'r').readlines()
questions = [q.strip() for q in questions if q.strip()]
intents = [i.strip() for i in intents if i.strip()]
# slot_index_file = open('slot_index.txt', 'w+')

unique_slots = open('unique_slot1.txt', 'r').read()
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
			key_write = key_write + '_' + key
		slots_dict[v] = key_write.strip()

# print(slots_dict)

for key in slots_dict.keys():
	print(key)
	print(slots_dict[key])

newlist = slots_dict.items()
sortedlist = sorted(newlist, key=lambda s: len(s[0]))
sortedlist.reverse()
# print(sortedlist)

# for qi in range(len(questions)):
# 	question = questions[qi]
# 	sentences = requests.post('http://localhost:9000/?properties={"annotators":"tokenize, ssplit, ner, pos", "outputFormat":"json"}', data=question).json()
	
# 	target_tokens = []
# 	for sentence in sentences['sentences']:
# 		for token in sentence['tokens']:
# 			print(token)

# 			if token['originalText'].find('LEXICA') > -1:
# 				t = token['originalText'].replace('LEXICA', '')
# 				t = t.split('_')
# 				target_tokens = target_tokens + t
# 			elif token['pos'] != '.':
# 				target_tokens.append('not')
# 	print(target_tokens)
# 	slot_index_file.write(' '.join(target_tokens) + '\n')

def slot_replacement(q, reg, label):
	ps = re.findall(reg, q)
	for tt in ps:
		item = ''.join(tt)
		print(item)
		count_space = [t for t in tt if t == ' ']
		replacement = [label for s in range(len(count_space) + 1)]
		replacement_string = ' '.join(replacement)
		q = q.replace(item, replacement_string)
	return q

filtered_questions = []
fitlered_slot_index = []
filtered_q_file = open('./kq.txt', 'w+')
filtered_s_file = open('./ks.txt', 'w+')
filtered_i_file = open('./k1i.txt', 'w+')
for qi in range(len(questions)):
	question = ' '+questions[qi].lower()+' '
	original_sentences = requests.post('http://localhost:9000/?properties={"annotators":"tokenize, ssplit, ner, pos", "outputFormat":"json"}', data=question.strip()).json()
	actual_tokens = []
	all_actual_tokens = []
	for sentence in original_sentences['sentences']:
		actual_sentence_tokens = []
		for token in sentence['tokens']:
			actual_sentence_tokens.append(token['originalText'])
			all_actual_tokens.append(token['originalText'])
		actual_tokens.append(actual_sentence_tokens)

	print(question)
	question = ' ' + ' '.join(all_actual_tokens) + ' '

	# Time Extractor
	question = slot_replacement(question, r" (\d{1,2})(:)(\d{1,2}) ", "LEXICATIME")
	question = slot_replacement(question, r" (\d{1,2})( )?(am|pm) ", "LEXICATIME")
	# Year Period Extractor
	question = slot_replacement(question, r" (19|20)?([0-9]{2})( )?(-)( )?(19|20)?([0-9]{2}) ", "LEXICAYEARPERIOD")
	# Time Period Extractor
	question = slot_replacement(question, r" ([1-9])([0-9])?( )?(-)( )?([1-9])([0-9])? ", "LEXICATIMEPERIOD")
	# Weekday Period Extractor
	question = slot_replacement(question, r" (mon|tue|tues|wed|thur|thurs|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)( )?(-)( )?(mon|tue|tues|wed|thur|thurs|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday) ", "LEXICAWEEKDAYPERIOD")
	# Day Extractor
	question = slot_replacement(question, r" (\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirdteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|few|couple)( )?(day|week|month|year)(s)? ", "LEXICANUMBEROFDAYS")
	# Library Material Extractor
	question = slot_replacement(question, r" (\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirdteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|few|couple)( )?(book|ebook|e-book|journal|ejournal|e-journal|cd|dvd|magazine|newspaper|movie|essay|dissertation|film|pamphlet)(s)? ", "LEXICALIBRARYMATERIAL")
	# Year Extractor
	question = slot_replacement(question, r" (19|20)([0-9]{2}) ", "YEAR")

	for item in sortedlist:
		if question.find(' '+item[0]+' ') > -1:
			question = question.replace(' '+item[0]+' ',  ' LEXICA'+item[1]+' ')
	sentences = requests.post('http://localhost:9000/?properties={"annotators":"tokenize, ssplit, ner, pos", "outputFormat":"json"}', data=question).json()
	if len(actual_tokens) == len(sentences['sentences']):
		for si in range(len(sentences['sentences'])):
			sentence = sentences['sentences'][si]
			target_tokens = []
			for token in sentence['tokens']:
				if token['originalText'].find('LEXICA') > -1:
					t = token['originalText'].replace('LEXICA', '')
					t = t.split('_')
					target_tokens = target_tokens + t
				else:
					target_tokens.append('not')
			# if len(list(set(target_tokens))) == 1 and list(set(target_tokens))[0] == "not":
			# 	print('not to be included.')
			# else:
			# 	if len(target_tokens) > 2:
			# 		at = [a.lower() for a  in actual_tokens[si]]
			# 		filtered_questions.append(' '.join(at))
			# 		fitlered_slot_index.append(' '.join(target_tokens))

			if 'TOPIC' in target_tokens:
				at = [a.lower() for a  in actual_tokens[si]]
				filtered_questions.append(' '.join(at))
				fitlered_slot_index.append(' '.join(target_tokens))
	else:
		print('Problem with tokenize')
		print(actual_tokens)

filtered_q_file.write('\n'.join(filtered_questions))
filtered_s_file.write('\n'.join(fitlered_slot_index))
filtered_i_file.write('\n'.join(intents))
