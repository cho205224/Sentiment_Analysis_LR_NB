import sys
import csv
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report

class POSTagExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([self.pos_tag_counts(doc) for doc in X])

    def pos_tag_counts(self, text):
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        pos_counts = Counter(tag for word, tag in pos_tags)
        tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
                'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 
                'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
        return np.array([pos_counts[tag] for tag in tags])
        
class TextLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([[len(doc)] for doc in X])

def preprocess_text(text):
    # Remove URL
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub('', text)

    # Remove Users
    " ".join(filter(lambda x:x[0]!='@', text.split()))

    # Convert to lowercase
    text = text.lower()

    # Remove repeating letters
    text = re.sub(r'(.)\1+', r'\1', text)

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]+', '', text)
    return text

def generate_labeled_corpus(inFile, inFile2):
    documents = []
    sentiments = []
    neutral = 0
    pos = 0
    neg = 0

    # Parse train.csv
    with open (inFile, 'r') as f:
        f.readline()
        reader = csv.reader(f)

        #line 0=id, 1=text, 2=selected, 3=sentiment, 4=time, 5=age, 6=country, 7=landArea, 8=density
        for line in reader:
            preprocess_text(line[2])
            documents.append(line[2])
            match line[3]:
                 case 'negative': # Negative
                      sentiments.append('0')
                      neg += 1
                 case 'neutral': # Neutral
                      sentiments.append('1')
                      neutral += 1
                 case 'positive': # Positive
                      sentiments.append('2')
                      pos += 1

    # Parse testdata.manual.2009.06.14.csv
    with open (inFile2, 'r') as f:
        reader = csv.reader(f)

        #line 0=sentiment, 1=id, 2=date, 3=query, 4=user, 5=txt
        for line in reader:
            preprocess_text(line[5])
            documents.append(line[5])
            match line[0]:
                 case '0': # Negative
                      sentiments.append('0')
                      neg += 1
                 case '2': # Neutral
                      sentiments.append('1')
                      neutral += 1
                 case '4': # Positive
                      sentiments.append('2')
                      pos += 1

    trainTest(documents, sentiments)

    # Data set sentiment size
    print(f'Negative : {neg}\nNeutral : {neutral}\nPositive : {pos}\n')

    return

def trainTest(corpus, labels):
    # TF-IDF with nltk stop words removal
    stop = list(stopwords.words('english'))
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop)

    # Split the data into training and testing sets, 20% test size
    X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=42)

    # Feature Union
    combined_features = FeatureUnion([
        ('tfidf', tfidf_vectorizer),
        ('text_length', TextLengthExtractor()),
        ('pos_tags', POSTagExtractor())
    ])

    # Perform logistic regression
    log_reg_pipeline = Pipeline([('features', combined_features),
                                 ('log_reg', LogisticRegression(max_iter=10000, solver='liblinear'))])

    # Define parameter grid
    param_grid = {#'log_reg__C': [0.01, 0.1, 1, 10, 100]
                  #'log_reg__C': [0.04, 0.7, 1, 1.3, 1.6]
                  #'log_reg__C': [1.1, 1.2, 1.3, 1.4, 1.5]
                  #'log_reg__C': [1.32, 1.36, 1.4, 1.44, 1.48]
                  #'log_reg__C': [1.37, 1.38, 1.39, 1.40, 1.41, 1.42, 1.43]
                  'log_reg__C': [1.41],  # Regularization
                  
                  #'log_reg__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                  'log_reg__solver': ['liblinear']  # Solver
                  }

    grid_search = GridSearchCV(log_reg_pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)

    # Train the logistic regression
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Predict on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    accuracy_log_reg = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy_log_reg}")
    print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'], zero_division=0))

    # Pipeline for Naive Bayes
    nb_pipeline = Pipeline([('features', combined_features)
                            ,('nb', MultinomialNB())])

    param_grid = {#'nb__alpha': [0.01, 0.1, 1, 10, 100]
                  #'nb__alpha': [0.04, 0.07, 0.1, 0.3, 0.4]
                  #'nb__alpha': [0.14, 0.22, 0.3, 0.34, 0.38]
                  #'nb__alpha': [0.16, 0.19, 0.22, 0.25, 0.28]
                  #'nb__alpha': [0.10, 0.13, 0.16, 0.19, 0.22]
                  'nb__alpha': [0.16]  # Alpha
                  }

    grid_search = GridSearchCV(nb_pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)
    
    # Train the Naive Bayes
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Predict on the test set using Naive Bayes
    y_pred = best_model.predict(X_test)

    # Evaluate the Naive Bayes
    accuracy_nb = accuracy_score(y_test, y_pred)
    print(f"Naive Bayes Accuracy: {accuracy_nb}")
    print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'], zero_division=0))

    return

if __name__ == '__main__':
        inFile = sys.argv[1]
        inFile2 = sys.argv[2]
        generate_labeled_corpus(inFile, inFile2)