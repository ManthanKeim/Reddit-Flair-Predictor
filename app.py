import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from nltk.corpus import stopwords
import sklearn
import pickle
import praw
import re
from bs4 import BeautifulSoup
import nltk

app = Flask(__name__, static_url_path='/static')
model = pickle.load(open('model.pkl', 'rb'))

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    
    text = BeautifulSoup(text, "lxml").text
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    reddit = praw.Reddit(client_id='V4ajjSi0MXlYDw', client_secret='4GrGn3IamlL5pLNvVFlk4GmZ1XQ', user_agent='research_pur')
    
    int_features = request.form['redditurl']
    processed_text = int_features.lower()
    print(processed_text)
    

    submission = reddit.submission(url=processed_text)
        
    data = {}

    data['title'] = submission.title
    data['title_un'] = submission.title
    data['url'] = submission.url
    data['selftext'] = submission.selftext

    submission.comments.replace_more(limit=None)
    comment = ''
    for top_level_comment in submission.comments:
        comment = comment + ' ' + top_level_comment.body
    data["comment"] = comment
    data['title'] = clean_text(data['title'])
    data['comment'] = clean_text(data['comment'])
    data['combine'] = data['title'] + data['comment'] + data['url']
    
    
#    prediction = model.predict(int_features)

    output = model.predict([data['combine']])
    
    return render_template('index.html', prediction_text='The flair for the subreddit post is : {}'.format(output))
    

@app.route('/about',methods=['POST'])
def about():
    
    reddit = praw.Reddit(client_id='V4ajjSi0MXlYDw', client_secret='4GrGn3IamlL5pLNvVFlk4GmZ1XQ', user_agent='research_pur')
    
    int_features = request.form['redditurl']
    processed_text = int_features.lower()
    print(processed_text)
    
    data = {}
    submission = reddit.submission(url=processed_text)
    
    data['title'] = submission.title
    data['title_un'] = submission.title
    data['url'] = submission.url
    data['selftext'] = submission.selftext
    
    return render_template('about.html', title = data['title_un'], About = data['selftext'])



if __name__ == "__main__":
    app.run(host="localhost", port=8090, debug=True)
