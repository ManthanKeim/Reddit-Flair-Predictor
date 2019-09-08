

import numpy as np
import pandas as pd
from textblob import TextBlob
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
import wordcloud
from collections import Counter
from pprint import pprint
import random
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

pd.options.display.float_format = '{:.2f}'.format
sns.set(style="ticks")
plt.rc('figure', figsize=(8, 5), dpi=100)
plt.rc('axes', facecolor="#ffffff", linewidth=0.4, grid=True, labelpad=8, labelcolor='#616161')
plt.rc('patch', linewidth=0)
plt.rc('xtick.major', width=0.2)
plt.rc('ytick.major', width=0.2)
plt.rc('grid', color='#9E9E9E', linewidth=0.4)
plt.rc('text', color='#282828')
plt.rc('savefig', pad_inches=0.3, dpi=300)

# Hiding warnings for cleaner display
warnings.filterwarnings('ignore')

# Configuring some notebook options
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
# If you want interactive plots, uncomment the next line
# %matplotlib notebook


India_df = pd.read_csv('reddit_07sep_2.csv')

subreddits = [India_df]

subreddit_names = ['India']

India_df.head()
print(India_df.info())

India_df['body'].fillna(value="", inplace=True)

def num_capitalized_word(s):
    c = 0
    for w in s.split():
        if w.isupper():
            c += 1
    return c

India_df['num_capitalized'] = India_df['title'].apply(num_capitalized_word)

contractions = {
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

# if a contraction has more than one possible expanded forms, we replace it
# with a list of these possible forms
tmp = {}
for k,v in contractions.items():
    if "/" in v:
        tmp[k] = [x.strip() for x in v.split(sep="/")]
    else:
        tmp[k] = v
contractions = tmp

tokenizer = RegexpTokenizer(r"[\w']+")
subreddit_words = []
for df, name in zip(subreddits, subreddit_names):
    all_titles = ' '.join([x.lower() for x in df['title']])
    for k,v in contractions.items():
        if isinstance(v, list):
            v = random.choice(v)
        all_titles = all_titles.replace(k.lower(), v.lower())
    words = list(tokenizer.tokenize(all_titles))
    words = [x for x in words if x not in stopwords.words('english')]
    subreddit_words.append(words)
    print('Most common words in ' + name, '*****************', sep='\n')
    pprint(Counter(words).most_common(35), compact=True)
    print()

subreddit_words_f = [x for y in subreddit_words for x in y]
print('Most common words in all subreddits', '*****************', sep='\n')
pprint(Counter(subreddit_words_f).most_common(35), compact=True)

def col_func(word, font_size, position, orientation, font_path, random_state):
    colors = ['#b58900', '#cb4b16', '#dc322f', '#d33682', '#6c71c4',
              '#268bd2', '#2aa198', '#859900']
    return random.choice(colors)

fd = {
    'fontsize': '32',
    'fontweight' : 'normal',
    'verticalalignment': 'baseline',
    'horizontalalignment': 'center',
}

for df, name, words in zip(subreddits, subreddit_names, subreddit_words):
    wc = wordcloud.WordCloud(width=1000, height=500, collocations=False,
                             background_color="#fdf6e3", color_func=col_func,
                             max_words=200,random_state=np.random.randint(1,8)
                             ).generate_from_frequencies(dict(Counter(words)))
    wc.to_file("india.png")
    fig, ax = plt.subplots(figsize=(20,10))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(name, pad=24, fontdict=fd)

for df in subreddits:
    df['title_length'] = df['title'].apply(lambda x: len(x))

fig = plt.subplots(figsize=(20,20), sharex=True, sharey=True)
#fig.subplots_adjust(hspace=0.5, wspace=0.4)
# fig.text(0.5, 0.04, 'Title Length', ha='center', fontdict={'size':'18'})
#for df, name in zip(subreddits, subreddit_names):
ax = sns.distplot(df["title_length"], kde=False, hist_kws={'alpha': 1}, color="#0747A6")

ax.set_title(name, fontdict={'size': 16}, pad=14)
ax.set(xlabel="Title length", ylabel="Number of posts")
plt.savefig("india2.png")

fig = plt.subplots(figsize=(20,20))
#fig.subplots_adjust(hspace=0.5, wspace=0.4)
# fig.text(0.5, 0.04, 'Number of Comments', ha='center', fontdict={'size':'18'})
# fig.text(0.04, 0.5, 'Number of Posts', va='center', rotation='vertical', fontdict={'size':'18'})
#for ax, df, name in zip(axes.flat, subreddits, subreddit_names):
ax = sns.distplot(df["comms_num"], kde=False, hist_kws={'alpha': 1}, color="#0747A6")
ax.set_title(name, fontdict={'size': 16}, pad=14)
ax.set(xlabel="Number of comments", ylabel="Number of posts")
plt.savefig("india7.png")

medians = []
for df in subreddits:
    medians.append(df['comms_num'].median())

plt.rc('axes', labelpad=16)
fig, ax = plt.subplots(figsize=(14,8))
d = pd.DataFrame({'subreddit': subreddit_names, 'comms_num_median': medians})
sns.barplot(x="subreddit", y="comms_num_median", data=d, palette=sns.cubehelix_palette(n_colors=24, reverse=True), ax=ax);
plt.savefig("india3.png")
ax.set(xlabel="Subreddit", ylabel="Median");
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
plt.rc('axes', labelpad=8)

fig = plt.subplots(figsize=(20,20))
#fig.subplots_adjust(hspace=0.5, wspace=0.4)
#for df, name in zip(subreddits, subreddit_names):
ax2 = sns.distplot(df["ups"], kde=False, hist_kws={'alpha': 0.5}, color="#0747A6")
ax2 = sns.distplot(df["downs"], kde=False, hist_kws={'alpha': 0.5}, color="#FF5630")

ax2.set_title(name, fontdict={'size': 16}, pad=14)
ax2.set(xlabel="Number of upvotes/downvotes", ylabel="Number of posts")
plt.savefig("india4.png")

comments = pd.DataFrame(columns=['subreddit', 'score'])
n = []
l = []
for df, name in zip(subreddits, subreddit_names):
    cl = list(df['score'])
    l.extend(cl)
    n.extend([name] * len(cl))
comments['subreddit'] = pd.Series(n)
comments['score'] = pd.Series(l)

fig, ax = plt.subplots(figsize=(20, 20))
sns.violinplot(x='score', y='subreddit', data=comments, scale='width', inner='box', ax=ax);
plt.savefig("india5.png")


fig = plt.subplots(figsize=(60,20))
#fig.subplots_adjust(hspace=0.7, wspace=0.3)
#for ax, df, name in zip(axes.flat, subreddits, subreddit_names):
ax3 = sns.heatmap(df[['score', 'ups', 'downs', 'comms_num', 'title_length', 'num_capitalized']].corr(), annot=True, cmap=sns.cubehelix_palette(as_cmap=True))

ax3.set_title(name, fontdict={'size':'18'}, pad=14)
ax3.set(xlabel="", ylabel="")
ax3.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.savefig("india6.png")

