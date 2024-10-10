import sys
import io
import re
import nltk
import string
nltk.download('punkt')       
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('omw-1.4')
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

text = pd.read_csv('train.txt', sep='\t', header=None, names=['Title', 'Country', 'Genre', 'Director', 'Plot'])
#list of plot strings
plot = text['Plot'].tolist()
    
stop_words = stopwords.words('english')
punc = string.punctuation
lemmatizer = WordNetLemmatizer()

new_tokens=[]
tokens = [nltk.word_tokenize(p) for p in plot]
tokens = [word.lower() for sublist in tokens for word in sublist]
lema = [lemmatizer.lemmatize(word) for word in tokens]
for token in lema:
    if token not in stop_words and token not in punc:
        new_tokens.append(token)

print(new_tokens)

#Dividir os dados em treino e validação

data = pd.read_csv('train.txt', sep='\t', header=None, names=['Title', 'Country', 'Genre', 'Director', 'Plot'])
genre_list = data['Genre'].tolist()

