import sys
import io
import re
import nltk
import string
nltk.download('punkt')       
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('train.txt', 'r', encoding='utf-8') as arquivo:
    conteudo = arquivo.read()
    
stop_words = stopwords.words('english')
punc = string.punctuation

new_tokens=[]
tokens=nltk.word_tokenize(conteudo)
for token in tokens:
    if token not in stop_words and token not in punc:
        new_tokens.append(token)

print(new_tokens)
