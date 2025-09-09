# Setup Environment and Import Libraries
!pip install streamlit

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier,
                              BaggingClassifier)
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('porter_test') # corrected from 'PorterStemmer' and 'punkt_tab'

# Load Dataset
df = pd.read_csv('/content/email_spam_detect_dataset.csv')
df.rename(columns={'email_text':'text', 'label':'target'}, inplace=True)
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Check missing and duplicated values
df.isnull().sum()
df.duplicated().sum()

# Exploratory Data Analysis (EDA)
df['target'].value_counts()
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct='%0.2f')
plt.show()

df['num_of_chars'] = df['text'].apply(len)
df['num_of_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_of_sent'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

sns.histplot(df[df['target'] == 0]['num_of_chars'])
sns.histplot(df[df['target'] == 1]['num_of_chars'], color='red')

sns.histplot(df[df['target'] == 0]['num_of_words'])
sns.histplot(df[df['target'] == 1]['num_of_words'], color='red')

sns.pairplot(df, hue='target')
sns.heatmap(df[['target', 'num_of_chars', 'num_of_words', 'num_of_sent']].corr(), annot=True)

# Data Preprocessing
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

df['transform_text'] = df['text'].apply(transform_text)

# WordCloud Visualization for spam and ham (Note: ham displayed erroneously as spam)
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='black')

spam_wc = wc.generate(df[df['target'] == 1]['transform_text'].str.cat(sep=" "))
plt.imshow(spam_wc)

ham_wc = wc.generate(df[df['target'] == 0]['transform_text'].str.cat(sep=" "))
plt.imshow(ham_wc)

# Most common words for spam and ham messages
from collections import Counter

spam_corpus = []
for msg in df[df['target'] == 1]['transform_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

ham_corpus = []
for msg in df[df['target'] == 0]['transform_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

# Plot of most common words
import matplotlib.pyplot as plt
import seaborn as sns

spam_word_freq = pd.DataFrame(Counter(spam_corpus).most_common(30))
sns.barplot(x=spam_word_freq[0], y=spam_word_freq[1])
plt.xticks(rotation='vertical')
plt.show()

ham_word_freq = pd.DataFrame(Counter(ham_corpus).most_common(30))
sns.barplot(x=ham_word_freq[0], y=ham_word_freq[1])
plt.xticks(rotation='vertical')
plt.show()

# Model Building and Evaluation

cv = CountVectorizer()
X = cv.fit_transform(df['transform_text']).toarray()
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# Train and evaluate models using CountVectorizer features
for clf, name in zip([gnb, mnb, bnb], ['GaussianNB', 'MultinomialNB', 'BernoulliNB']):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"{name} Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")
    print(f"{name} Precision: {precision_score(y_test, y_pred)}")

# Repeat using TfidfVectorizer
tf = TfidfVectorizer()
X = tf.fit_transform(df['transform_text']).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

for clf, name in zip([gnb, mnb, bnb], ['GaussianNB', 'MultinomialNB', 'BernoulliNB']):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{name} Accuracy with TF-IDF: {accuracy_score(y_test, y_pred)}")
    print(f"{name} Confusion Matrix with TF-IDF:\n {confusion_matrix(y_test, y_pred)}")
    print(f"{name} Precision with TF-IDF: {precision_score(y_test, y_pred)}")

# Since all models perform similarly, proceed with TfidfVectorizer + MultinomialNB

# Additional classifiers setup
svc = SVC(kernel='linear', C=1.0, random_state=2)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC': svc,
    'KN': knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'ABC': abc,
    'BC': bc,
    'ETC': etc,
    'GBDT': gbdt,
    'XGB': xgb
}

def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision

for name, clf in clfs.items():
    accuracy, precision = train_classifier(clf, X_train, y_train, X_test, y_test)
    print(f"For {name} -> Accuracy: {accuracy}, Precision: {precision}")

performance_def = pd.DataFrame({
    'Algorithm': list(clfs.keys()),
    'Accuracy': [train_classifier(clf, X_train, y_train, X_test, y_test)[0] for clf in clfs.values()],
    'Precision': [train_classifier(clf, X_train, y_train, X_test, y_test)[1] for clf in clfs.values()]
}).sort_values('Precision', ascending=False)

sns.catplot(x='Algorithm', y='Precision', data=performance_def, kind='bar', height=5)
plt.title('Precision Scores')
plt.ylim(0.5, 1.0)
plt.xticks(rotation='vertical')
plt.show()







