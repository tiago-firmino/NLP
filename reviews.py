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
    
    # Naive Bayes + CountVectorizer Pipeline
    pipeline_nb = Pipeline([
        ('vect', CountVectorizer(max_features=500, stop_words='english', max_df=0.7)),
        ('clf', MultinomialNB())
    ])
    
    # Cross-validation for Naive Bayes
    scores_nb = perform_cross_validation(pipeline_nb, X, y, skf)
    print(f"Naive Bayes Cross-Validation Accuracy: {scores_nb.mean():.4f} ± {scores_nb.std():.4f}")
    
    # SVC + TF-IDF Vectorizer Pipeline
    pipeline_svc = Pipeline([
        ('vect', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('clf', SVC(kernel='linear', C=1.0))
    ])
    
    # Perform cross-validation for SVC
    scores_svc = perform_cross_validation(pipeline_svc, X, y, skf)
    print(f"SVC Cross-Validation Accuracy: {scores_svc.mean():.4f} ± {scores_svc.std():.4f}")
    
    # Logistic Regression + TF-IDF Pipeline
    pipeline_lr = Pipeline([
        ('vect', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    # Perform cross-validation for Logistic Regression
    scores_lr = perform_cross_validation(pipeline_lr, X, y, skf)
    print(f"Logistic Regression Cross-Validation Accuracy: {scores_lr.mean():.4f} ± {scores_lr.std():.4f}")
    
    # Optionally, you can choose the best model based on cross-validation scores
    # For demonstration, let's proceed with Logistic Regression as before
    
    # Split into train and test sets for final evaluation (optional)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression Model on the full training set
    pipeline_lr.fit(X_train, y_train)
    y_pred_lr = pipeline_lr.predict(X_test)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
    print(f"Logistic Regression Final Test Accuracy: {accuracy_lr}")
    
    # Load and preprocess the test data (test_no_labels.txt) with all fields
    X_test_no_labels = load_test_data('test_no_labels.txt')
    X_test_no_labels = preprocess_text(X_test_no_labels)
    
    # Use the best-performing model (SVC) to predict genres for the test set
    predictions = pipeline_svc.predict(X_test_no_labels)
    
    # Save the results to results.txt
    with open('results.txt', 'w') as f:
        for genre in predictions:
            f.write(f"{genre}\n")
