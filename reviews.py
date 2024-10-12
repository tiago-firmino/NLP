## PROJECT CODE GROUP 21
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
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

# Naive Bayes + CountVectorizer
def train_naive_bayes(X_train, y_train):
    count_vect = CountVectorizer(max_features=500, stop_words='english', max_df=0.7)
    X_train_counts = count_vect.fit_transform(X_train)
    clf_nb = MultinomialNB()
    clf_nb.fit(X_train_counts, y_train)
    return clf_nb, count_vect

# SVC + TF-IDF Vectorizer«
def train_svc(X_train, y_train):
    tfidf_vect = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = tfidf_vect.fit_transform(X_train)
    clf_svc = SVC(kernel='linear', C=1.0)
    clf_svc.fit(X_train_tfidf, y_train)
    return clf_svc, tfidf_vect

# Logistic Regression + TF-IDF
def train_logistic_regression(X_train, y_train):
    tfidf_vect = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = tfidf_vect.fit_transform(X_train)
    clf_lr = LogisticRegression(max_iter=1000)
    clf_lr.fit(X_train_tfidf, y_train)
    return clf_lr, tfidf_vect

# Predict and evaluate the model
def evaluate_model(clf, vectorizer, X_test, y_test):
    X_test_transformed = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, conf_matrix

# Main workflow
if __name__ == "__main__":
    # Load and preprocess data
    X, y = load_data('train.txt')
    X = preprocess_text(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Naive Bayes Model
    clf_nb, count_vect = train_naive_bayes(X_train, y_train)
    accuracy_nb, conf_matrix_nb = evaluate_model(clf_nb, count_vect, X_test, y_test)
    print(f"Naive Bayes Accuracy: {accuracy_nb}")
    
    # Train SVC Model
    clf_svc, tfidf_vect_svc = train_svc(X_train, y_train)
    accuracy_svc, conf_matrix_svc = evaluate_model(clf_svc, tfidf_vect_svc, X_test, y_test)
    print(f"SVC Accuracy: {accuracy_svc}")
    
    # Train Logistic Regression Model
    clf_lr, tfidf_vect_lr = train_logistic_regression(X_train, y_train)
    accuracy_lr, conf_matrix_lr = evaluate_model(clf_lr, tfidf_vect_lr, X_test, y_test)
    print(f"Logistic Regression Accuracy: {accuracy_lr}")

    # Load and preprocess the test data (test_no_labels.txt) with all fields
    X_test_no_labels = load_test_data('test_no_labels.txt')
    X_test_no_labels = preprocess_text(X_test_no_labels)

    # Use the best-performing model to predict genres for the test set
    predictions = clf_lr.predict(tfidf_vect_lr.transform(X_test_no_labels))
    
    # Save the results to results.txt
    with open('results.txt', 'w') as f:
        for genre in predictions:
            f.write(f"{genre}\n")
