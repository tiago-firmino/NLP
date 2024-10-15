## PROJECT CODE GROUP 21
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load data with all relevant fields
def load_data(filepath):
    # Expecting a TSV file (train.txt) with columns: title, from, genre, director, plot
    data = pd.read_csv(filepath, sep='\t', names=['title', 'from', 'genre', 'director', 'plot'])
    # Combine relevant fields into one string
    data['text'] = data['title'] + ' ' + data['from'] + ' ' + data['director'] + ' ' + data['plot']
    return data['text'], data['genre']

# Load and preprocess test set (test_no_labels.txt) using all fields
def load_test_data(filepath):
    test_data = pd.read_csv(filepath, sep='\t', names=['title', 'from', 'director', 'plot'])
    test_data['text'] = test_data['title'] + ' ' + test_data['from'] + ' ' + test_data['director'] + ' ' + test_data['plot']
    return test_data['text']

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()

    text = text.str.replace(r'[^\w\s]', '', regex=True)
    text = text.str.lower()

    stop_words = set(stopwords.words('english'))

    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    text = text.apply(lambda x: x.split())
    text = text.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    text = text.apply(lambda x: ' '.join(x))
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return text

# Cross-validation function
def perform_cross_validation(pipeline, X, y, cv):
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores

# Main workflow
if __name__ == "__main__":
    # Load and preprocess data
    X, y = load_data('train.txt')
    X = preprocess_text(X)
    
    # Cross-validation strategy
    k = 10  # Number of folds
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # SVC + TF-IDF Vectorizer Pipeline
    pipeline_svc = Pipeline([
        ('vect', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('clf', SVC(kernel='linear', C=1.0))
    ])
    
    # Perform cross-validation for SVC
    scores_svc = perform_cross_validation(pipeline_svc, X, y, skf)
    print(f"SVC Cross-Validation Accuracy: {scores_svc.mean():.4f} ± {scores_svc.std():.4f}")
    
    # Load and preprocess the test data (test_no_labels.txt) with all fields
    X_test_no_labels = load_test_data('test_no_labels.txt')
    X_test_no_labels = preprocess_text(X_test_no_labels)

    pipeline_svc.fit(X, y)
    
    # Use the best-performing model (SVC) to predict genres for the test set
    predictions = pipeline_svc.predict(X_test_no_labels)
    
    # Save the results to results.txt
    with open('results.txt', 'w') as f:
        for genre in predictions:
            f.write(f"{genre}\n")
