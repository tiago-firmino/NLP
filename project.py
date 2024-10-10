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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('train.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    content = content.lower()
    
stop_words = stopwords.words('english')
punc = string.punctuation
lemmatizer = WordNetLemmatizer()

new_tokens=[]
tokens=nltk.word_tokenize(content)
lemm = [lemmatizer.lemmatize(word) for word in tokens]
for token in tokens:
    if token not in stop_words and token not in punc:
        new_tokens.append(token)

print(new_tokens)
