from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
import pickle

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return mnb, accuracy, precision

def save_model(model, vectorizer, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_model(model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer
