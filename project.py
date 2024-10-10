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

with open('train.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    content = content.lower()
    
stop_words = stopwords.words('english')
punc = string.punctuation
lemmatizer = WordNetLemmatizer()

new_tokens=[]
tokens=nltk.word_tokenize(content)
lema = [lemmatizer.lemmatize(word) for word in tokens]
for token in lema:
    if token not in stop_words and token not in punc:
        new_tokens.append(token)

#Dividir os dados em treino e validação

data = pd.read_csv('train.txt', sep='\t', header=None, names=['Title', 'Country', 'Genre', 'Director', 'Plot'])
genre_list = data['Genre'].tolist()

print(genre_list)
