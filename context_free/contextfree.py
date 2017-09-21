from nltk.parse.generate import generate, demo_grammar
from nltk import CFG
from io import open
grammar = CFG.fromstring(""" 
    S -> NP PP Topic | VP NP PP Topic | Ask VP NP PP Topic
    NP -> N | Det N
    Ask -> 'please' | 'can' Person |
    VP -> V | V2 'me' V
    PP -> 'about'| 'on' | 'for' | 'regarding' | 'from'
    N -> 'books'|'articles'|'journals'|'book'
    Det -> 'the'|'a'|'some'
    Person -> 'you'
    V -> 'find'
    V2 -> 'help'
    Topic -> 'architecture' |'maths' | 'math' | 'calculus' | 'art' | 'arts' | 'economics' | 'econ' | 'politics' | 'commerce' | 'computer_science' | 'engineering' | 'music' | 'dance' | 'english' | 'chinese' | 'biology' | 'postmodern_architecture'
""")
print(grammar)
file=open('./sentences2.txt','ab')
for sentence in generate(grammar):
        file.write(' '.join(sentence))
        file.write('\n')
