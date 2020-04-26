# Reddit Flair Predictor

[Project Hosted on : Heroku](https://damp-tundra-68404.herokuapp.com)

[Machine Learning Model: Google Drive link](https://drive.google.com/file/d/1C_sxTrOCIoiqIPBieokMOYtD-JxmJCgE/view?usp=sharing)


### Technologies Used:
* Frontend - HTML/CSS
* Backend - Flask
* Database - Direct CSV import from pandas library
* APIs - Reddit API, PRAW Model 

### Overall Approach towards the problem

1. Data Acquisition - I collected the data from the subreddit r/India, using the reddit API and praw model. I collected almost 200 post from each flair incline (AskIndia,Non-Political,[R]eddiquette, Scheduled, Photography, Science/Technology, Politics, Business/Finance, Policy/Economy, Sports, Food, AMA). I collected approximately 2200 subreddit posts collectively and represented them using graphs, wordcloud etc. ( attached below).

2. Flair Detection - Since I'm not so fluent in Machine Learning part, but I'm able to make the model using different algorithms including Naive Bayes, SVM, logistic regression, random forest, MLP classifier. I got the best accuracy from random forest and using this as the testing and training. Then I've split the data into 70% training and 30% testing and getting the Random forest accuracy 78% using the combination of URL, comments and title of the subreddit post. (refrences attached ) 

3. Web Application : Using Flask library as backend and HTML/CSS as frontend, all the screenshots are attached below. Unfortunately I am not able to push the CSS file to heroku library due to memory shortage ( working on this ) but the application working is totally fine.

4. Reported the result using graphs and visualizations.

### Project Screenshots on LocalHost

<img alt="image_1" src="working_proj/Screenshot 2019-09-08 at 3.55.44 PM.png" width="700px">

Starting Screen 

<img alt="image_1" src="working_proj/Screenshot 2019-09-08 at 3.55.33 PM.png" width="700px">

Predicted Flair

<img alt="image_1" src="working_proj/Screenshot 2019-09-08 at 3.56.54 PM.png" width="700px">

About Subreddit


### Data Analysis with collected data

<img alt="image_1" src="working_proj/india.png" width="700px">

WordCloud

<img alt="image_1" src="working_proj/india2.png" width="700px">

Title Length

<img alt="image_1" src="working_proj/india7.png" width="700px">

Comment Length

<img alt="image_1" src="working_proj/india4.png" width="700px">

Number of Upvotes/Downvotes

<img alt="image_1" src="working_proj/india8.png" width="700px">

<img alt="image_1" src="working_proj/india5.png" width="700px">

Distribution of Score

<img alt="image_1" src="working_proj/india6.png" width="1000px">

Correlation Heatmap

### Libraries/Dependencies

* beautifulsoup
* Flask
* scikit
* sklearn
* nltk
* etc. ( listed in requirements.txt)

### Refrences

* https://praw.readthedocs.io/en/latest/
* https://www.reddit.com/dev/api/
* https://www.datacamp.com/community/tutorials/wordcloud-python
* https://seaborn.pydata.org

R
