# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

flairs = ["AskIndia", "Non-Political", "[R]eddiquette", "Scheduled", "Photography", "Science/Technology", "Politics", "Business/Finance", "Policy/Economy", "Sports", "Food", "AMA"]


dataset = pd.read_csv('reddit_07sep_2.csv')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def string_form(value):
    return str(value)

def clean_text(text):
    
    text = BeautifulSoup(text, "lxml").text
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

india_csv = pd.read_csv('data.csv')
india_csv.head(10)

india_csv['title'] = india_csv['title'].apply(string_form)
india_csv['body'] = india_csv['body'].apply(string_form)
india_csv['comments'] = india_csv['comments'].apply(string_form)

india_csv['title'] = india_csv['title'].apply(clean_text)
india_csv['body'] = india_csv['body'].apply(clean_text)
india_csv['comments'] = india_csv['comments'].apply(clean_text)

feature_combine = india_csv["title"] + india_csv["comments"] + india_csv["url"]
india_csv = india_csv.assign(feature_combine = feature_combine)


def nb_classifier(X_train, X_test, y_train, y_test):
    
    from sklearn.naive_bayes import MultinomialNB
        
        
    nb = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', MultinomialNB()),
                       ])
    nb.fit(X_train, y_train)
                           
    y_pred = nb.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=flairs))



def linear_svm(X_train, X_test, y_train, y_test):
    
    from sklearn.linear_model import SGDClassifier
        
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, y_train)
                        
    y_pred = sgd.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=flairs))


def logisticreg(X_train, X_test, y_train, y_test):
    
    from sklearn.linear_model import LogisticRegression
        
    logreg = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                           ])
    logreg.fit(X_train, y_train)
                               
    y_pred = logreg.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=flairs))




def randomforest(X_train, X_test, y_train, y_test):
    
    from sklearn.ensemble import RandomForestClassifier
        
    ranfor = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('clf', RandomForestClassifier(n_estimators = 1000, random_state = 42)),
                           ])
    ranfor.fit(X_train, y_train)
    pickle.dump(ranfor, open('model.pkl','wb'))
    y_pred = ranfor.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=flairs))



def mlpclassifier(X_train, X_test, y_train, y_test):
    
    from sklearn.neural_network import MLPClassifier
        
    mlp = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MLPClassifier(hidden_layer_sizes=(30,30,30))),
                        ])
    mlp.fit(X_train, y_train)
                            
    y_pred = mlp.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=flairs))

def train_test(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

    print("Results of Naive Bayes Classifier")
    nb_classifier(X_train, X_test, y_train, y_test)
    print("Results of Linear Support Vector Machine")
    linear_svm(X_train, X_test, y_train, y_test)
    print("Results of Logistic Regression")
    logisticreg(X_train, X_test, y_train, y_test)
    print("Results of Random Forest")
    randomforest(X_train, X_test, y_train, y_test)
    print("Results of MLP Classifier")
    mlpclassifier(X_train, X_test, y_train, y_test)

cat = india_csv.flair

V = india_csv.feature_combine
W = india_csv.comments
X = india_csv.title
Y = india_csv.body
Z = india_csv.url

print("Flair Detection using Title as Feature")
#train_test(X,cat)
print("Flair Detection using Body as Feature")
#train_test(Y,cat)
print("Flair Detection using URL as Feature")
#train_test(Z,cat)
print("Flair Detection using Comments as Feature")
train_test(V,cat)


#dataset['experience'].fillna(0, inplace=True)
#
#dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

#X = dataset.iloc[:, :3]

#Converting words to integer values
#def convert_to_int(word):
#    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
#                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
#    return word_dict[word]

#X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

#y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()

#Fitting model with trainig data
#regressor.fit(X, y)

# Saving model to disk


# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))
