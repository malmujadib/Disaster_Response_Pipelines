# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle


warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    load_data function: It loads data from database, by using database path as parmater, 
    Input: Database path
    Output: Two dataframes, one for features, the other one for targets, also categories name
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))   
    df=pd.read_sql_query('SELECT * FROM DisasterResponse',engine)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace = True)
    X = df['message']
    Y = df.drop(['message','original','genre','id'], axis = 1)
    category_names = df.drop(columns = ['id', 'message', 'original', 'genre']).columns.values
    
    return X, Y, category_names

def tokenize(text):

    """
    tokenize function apply text tokenization to whole dataset that contains texts, by making it lower case, then remove punctuation, then tokenize each word, then removing stop words, and lastly lemmatize the text
    Input: Raw text
    Output: Text after processing
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text = word_tokenize(text)
    text = [w for w in text if w not in stopwords.words('english')]
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    return text


def build_model():

    """
    Build_model function build up the Logistic Regression model using Pipleine, and GridSearch
    Input: None
    Output: Logistic Regression Model
    """
    LR=LogisticRegression(max_iter=10000)

        # Pipline that include Tfidf and OVR
    pipeline = Pipeline([('tfidf', TfidfVectorizer(tokenizer=tokenize)), ('OvR', OneVsRestClassifier(LR))])

    parameters = {
            'OvR__estimator__C': [0.1, 1, 10, 100, 1000]
        }   
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    """
    evaluate_model funciton printout report about perfomance of with model
    Input: Machine Learning Model, the whole test set, and the category names
    Output: Print about each metric in the classifier
    """
    y_pred = model.predict(X_test)

    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):

    """
    save_model function, it saves the model as (pkl)
    Input: Machine Learning model, and the model path
    Output: Saved model as (pkl)
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():

    """
    main function triggers all functions to do it function
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()