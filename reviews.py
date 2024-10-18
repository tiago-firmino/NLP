import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def load_data(filepath):
    data = pd.read_csv(filepath, sep='\t', names=['title', 'from', 'genre', 'director', 'plot'])
    data['text'] = data['title'] + ' ' + data['from'] + ' ' + data['director'] + ' ' + data['plot']
    return data['text'], data['genre']

def load_test_data(filepath):
    test_data = pd.read_csv(filepath, sep='\t', names=['title', 'from', 'director', 'plot'])
    test_data['text'] = test_data['title'] + ' ' + test_data['from'] + ' ' + test_data['director'] + ' ' + test_data['plot']
    return test_data['text']

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.str.replace(r'[^\w\s]', '', regex=True).str.lower()
    stop_words = set(stopwords.words('english'))
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    text = text.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    return text.str.replace(r'\s+', ' ', regex=True).str.strip()

def train_naive_bayes(X_train, y_train):
    count_vect = CountVectorizer(max_features=10000, stop_words='english', max_df=0.7)
    X_train_counts = count_vect.fit_transform(X_train)
    clf_nb = MultinomialNB()
    clf_nb.fit(X_train_counts, y_train)
    return clf_nb, count_vect

def train_svc(X_train, y_train):
    tfidf_vect = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = tfidf_vect.fit_transform(X_train)
    clf_svc = SVC(kernel='linear', C=1.0, probability=True)
    clf_svc.fit(X_train_tfidf, y_train)
    return clf_svc, tfidf_vect

def train_logistic_regression(X_train, y_train):
    tfidf_vect = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = tfidf_vect.fit_transform(X_train)
    clf_lr = LogisticRegression(max_iter=1000)
    clf_lr.fit(X_train_tfidf, y_train)
    return clf_lr, tfidf_vect

def evaluate_model(clf, vectorizer, X_test, y_test):
    X_test_transformed = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, conf_matrix

# Combine model predictions based on weighted probabilities
def weighted_vote_proba(models, vectorizers, genre_accuracies, X_test):
    # Convert X_test to a list to ensure proper indexing
    X_test = list(X_test)
    
    final_predictions = []

    for i in range(len(X_test)):
        genre_scores = {}

        # For each model, use predict_proba to get the class probabilities
        for model_name, (clf, vectorizer) in models.items():
            proba = clf.predict_proba(vectorizer.transform([X_test[i]]))[0]

            # Get accuracy weights for the current model
            for genre, prob in zip(clf.classes_, proba):
                weighted_score = prob * genre_accuracies[model_name].get(genre, 0)
                if genre not in genre_scores:
                    genre_scores[genre] = 0
                genre_scores[genre] += weighted_score

        # Select the genre with the highest score
        final_prediction = max(genre_scores, key=genre_scores.get)
        final_predictions.append(final_prediction)

    return final_predictions

def evaluate_weighted_vote_accuracy(final_predictions, y_test):
    accuracy = accuracy_score(y_test, final_predictions)
    return accuracy

# Calculate confusion matrix for weighted voting system and plot
def calculate_conf_matrix_for_weighted_vote(final_predictions, y_test, genres):
    conf_matrix = confusion_matrix(y_test, final_predictions, labels=genres)
    return conf_matrix

# Calculate accuracy per genre
def calculate_genre_accuracies(conf_matrix, genres):
    return conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# Print a simple table for the accuracies per genre
def print_genre_accuracies_table(accuracies, weighted_genre_accuracies, genres):
    print(f"{'Genre':<15} {'Naive Bayes':<15} {'SVC':<15} {'Logistic Regression':<20} {'Weighted Voting':<15}")
    for i, genre in enumerate(genres):
        print(f"{genre:<15} {accuracies['Naive Bayes'][genre]:<15.2f} {accuracies['SVC'][genre]:<15.2f} {accuracies['Logistic Regression'][genre]:<20.2f} {weighted_genre_accuracies[i]:<15.2f}")

if __name__ == "__main__":
    X, y = load_data('train.txt')
    X = preprocess_text(X)

    # Split into train and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train models
    clf_nb, count_vect = train_naive_bayes(X_train, y_train)
    clf_svc, tfidf_vect_svc = train_svc(X_train, y_train)
    clf_lr, tfidf_vect_lr = train_logistic_regression(X_train, y_train)

    # Evaluate models (Naive Bayes, SVC, Logistic Regression)
    accuracy_nb, conf_matrix_nb = evaluate_model(clf_nb, count_vect, X_test, y_test)
    accuracy_svc, conf_matrix_svc = evaluate_model(clf_svc, tfidf_vect_svc, X_test, y_test)
    accuracy_lr, conf_matrix_lr = evaluate_model(clf_lr, tfidf_vect_lr, X_test, y_test)

    print(f"Naive Bayes Accuracy: {accuracy_nb}")
    print(f"SVC Accuracy: {accuracy_svc}")
    print(f"Logistic Regression Accuracy: {accuracy_lr}")

    # Calculate accuracy per genre
    genres = clf_nb.classes_
    genre_accuracies_nb = conf_matrix_nb.diagonal() / conf_matrix_nb.sum(axis=1)
    genre_accuracies_svc = conf_matrix_svc.diagonal() / conf_matrix_svc.sum(axis=1)
    genre_accuracies_lr = conf_matrix_lr.diagonal() / conf_matrix_lr.sum(axis=1)

    # Prepare model and accuracy data for weighted voting
    models = {
        'Naive Bayes': (clf_nb, count_vect),
        'SVC': (clf_svc, tfidf_vect_svc),
        'Logistic Regression': (clf_lr, tfidf_vect_lr)
    }
    
    accuracies = {
        'Naive Bayes': dict(zip(genres, genre_accuracies_nb)),
        'SVC': dict(zip(genres, genre_accuracies_svc)),
        'Logistic Regression': dict(zip(genres, genre_accuracies_lr))
    }

    predictions = weighted_vote_proba(models, 
                                      {'Naive Bayes': count_vect, 'SVC': tfidf_vect_svc, 'Logistic Regression': tfidf_vect_lr}, 
                                      accuracies, X_test)

    # Evaluate and print the weighted voting accuracy
    weighted_accuracy = evaluate_weighted_vote_accuracy(predictions, y_test)
    print(f"Weighted Voting Accuracy: {weighted_accuracy}")

    # Use the weighted voting system on the test data
    X_test_no_labels = load_test_data('test_no_labels.txt')
    X_test_no_labels = preprocess_text(X_test_no_labels)
    
    # Get final predictions from the weighted voting system
    final_predictions = weighted_vote_proba(models, 
                                      {'Naive Bayes': count_vect, 'SVC': tfidf_vect_svc, 'Logistic Regression': tfidf_vect_lr}, 
                                      accuracies, X_test_no_labels)
    
    # Write the final predictions to results.txt
    with open('results.txt', 'w') as f:
        for genre in final_predictions:
            f.write(f"{genre}\n")