from io import open
from spell import correction
import re
# p = inflect.engine()
def user_question():
    # question=input("Enter question\n")
    pattern='["A-Za-z]'
    m=re.compile(pattern)
    with open('questions_new.txt','r') as file:
        file2=open('intents.txt','r')
        for question,intent in zip(file,file2):
            if not m.match(question):
                continue
            if question[0]=='\"':
                q=next(file)
                question=question+" "+q
                intent=next(file2)
                while '\"' not in q:
                    question=question+" "+q
                    q=next(file)
                    intent=next(file2)
            print(question)
            print("INTENT = %s" %(intent))
            question=question.replace('\n', ' ').replace('\r', '')
            question=pre_proc(question)
            print('*************')

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
    sentence=sentence.replace('\n',' ')

    # sentence=sentence.replace(',','')
    sentence=sentence.lower()
    phrases=['_excuse_me_','_please_','_help_me_','_can_you_','_hi_','_hello_','_hey_','_good_morning_','_good_evening_','_may_i_','_kindly_','_know_','_can_i_','_should_i_','_i\'m_','_am_','_are_','_was_','_is_','_do_','_did_','_does_','_of_the_','_for_the_','_i_am_']
    phrases=sorted(phrases,reverse=True,key=len)
    temp="_"+sentence.replace(' ','_')+"_"
    print(temp)
    for phrase in phrases:
        if phrase in temp:
            temp=temp.replace(phrase,' ')
    sentence=temp.replace('__','_')
    sentence=sentence.replace('_',' ')
    # for word in sentence:
    # 	sing=p.singular_noun(word)
    # 	if sing!=word:
    # 		sentence.replace(word,sing)


    sentence=str(sentence[1].upper())+sentence[2:].lower()
    sentence=sentence.strip(' ')
    # corr=[correction(word) for word in sentence]
    # sentence=" ".join(corr)
    # sentence=sentence.strip(string.punctuation())
    print(sentence)
    return sentence
import re
pattern='["A-Za-z]'
m=re.compile(pattern)
with open('questions_new.txt','r') as file:
    file2=open('intents.txt','r')
    for question,intent in zip(file,file2):
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
        print(question)
        print("INTENT = %s" %(intent))
        question=question.replace('\n', ' ').replace('\r', '')
        questions.append(question)
        intents.append(intent)

questions = [s.strip() for s in questions if s.strip()]
intents = [s.strip() for s in intents if s.strip()]
print(list(zip(questions,intents)))