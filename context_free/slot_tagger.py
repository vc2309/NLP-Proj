from io import open
import nltk
subjects=['architecture','maths' , 'math' , 'calculus' , 'art' , 'arts' , 'economics' , 'econ' , 'politics' , 'commerce' , 'computer science' , 'engineering' , 'music' , 'dance' , 'english' , 'chinese' , 'biology' , 'postmodern architecture']
librarymaterials=['book','books','articles','journals']
librarystaff=['you']

f=open('./sentences2.txt','rb')
f2=open('./slots.txt','ab')
for line in f:
    l=nltk.word_tokenize(line)
    for word in l:
        print word
        if word in subjects:
            f2.write('SUBJECT ')
        elif word in librarymaterials:
            f2.write('LIBRARYMATERIALS ')
        elif word in librarystaff:
            f2.write('LIBRARYSTAFF ')
        else:
            f2.write('not ')
    f2.write('\n')

