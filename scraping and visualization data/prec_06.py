import praw
import pandas as pd
import csv
from time import sleep
from os.path import isfile
import datetime as dt

reddit = praw.Reddit(client_id='V4ajjSi0MXlYDw', client_secret='4GrGn3IamlL5pLNvVFlk4GmZ1XQ', user_agent='research_pur')

flairs = ["AskIndia", "Non-Political", "[R]eddiquette", "Scheduled", "Photography", "Science/Technology", "Politics", "Business/Finance", "Policy/Economy", "Sports", "Food", "AMA"]

subreddit = reddit.subreddit('india')
topics_dict = {"flair":[], "title":[], "score":[], "id":[], "url":[], "comms_num": [], "created": [], "body":[], "author":[], "comments":[], "ups":[], "downs":[]}

def get_date(created):
    return dt.datetime.fromtimestamp(created)


for flair in flairs:
    
    get_subreddits = subreddit.search(flair, limit=200)

    for submission in get_subreddits:

        topics_dict["flair"].append(flair)
        topics_dict["title"].append(submission.title)
        topics_dict["score"].append(submission.score)
        topics_dict["id"].append(submission.id)
        topics_dict["url"].append(submission.url)
        topics_dict["comms_num"].append(submission.num_comments)
        topics_dict["created"].append(submission.created)
        topics_dict["body"].append(submission.selftext)
        topics_dict["author"].append(submission.author)
        topics_dict["ups"].append(submission.ups)
        
        
        ratio = submission.upvote_ratio
    
        ups = round((ratio*submission.score)/(2*ratio - 1)) if ratio != 0.5 else round(submission.score/2)
        downs = ups - submission.score
        topics_dict["downs"].append(downs)

        submission.comments.replace_more(limit=None)
        comment = ''
        for top_level_comment in submission.comments:
            comment = comment + ' ' + top_level_comment.body
        topics_dict["comments"].append(comment)

topics_data = pd.DataFrame(topics_dict)
_timestamp = topics_data["created"].apply(get_date)
topics_data = topics_data.assign(timestamp = _timestamp)
del topics_data['created']
topics_data.to_csv('reddit_07sep_2.csv', index=False)
