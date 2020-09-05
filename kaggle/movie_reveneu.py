import sys
import json
import re
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
#import xgboost as xgb
#import lightgbm as lgb
#import catboost as cat
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_regression
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from math import sqrt
traindata='train.csv'
import re

reap = re.compile(r"([a-zA-Z]'[a-zA-Z])")

def words_clearing(words, remove_hash = True):
    new_words = []
    for word in words :
        word = word.lower().strip().replace('_', '')
        if remove_hash : word = word.replace('#', ' ')
        word = re.sub(r'([\W])+$', '', word)
        word = re.sub(r'^([\W])+', '', word)
        word = re.sub(r"[\W][st]$", '', word)
        if re.search(r'^http', word) or re.search(r'[^a-z]', word) or len(word)<2 :
            continue
        word = re.sub(r'[\W]', '', word)
        new_words.append(word)
    return new_words

def make_bag(tweet) :
    try :
        words = tweet['text'].split(' ')
    except KeyError :
        return []
    words = words_clearing(words)
    return words

def update_vocabulary(words, vocabulary):
    for word in words :
        if word in vocabulary :
            vocabulary[word] += 1
        else :
            vocabulary[word] = 1
    return vocabulary

def calculate_total_words(vocabulary) :
    sum = 0
    for word in vocabulary :
        sum += vocabulary[word]
    return sum

def calculate_frequency(n, total) :
    f = float(n)/float(total)
    return f

def make_scores(sent_file) :
    scores = {}
    for line in sent_file:
        term, score = line.split("\t")
        scores[term] = int(score)
    return scores

def neg_assess(tweet, scores):
    try :
        words = tweet.split(' ')
    except KeyError :
        return 0
    neg_sentiment = 0
    words = words_clearing(words)
    for word in words :
        if word in scores :
            if scores[word] < 0 :
                neg_sentiment += -1*scores[word]
    return neg_sentiment

def pos_assess(tweet, scores):
    try :
        words = tweet.split(' ')
    except KeyError :
        return 0
    pos_sentiment = 0
    words = words_clearing(words)
    for word in words :
        if word in scores :
            if scores[word] > 0 :
                pos_sentiment += scores[word]
    return pos_sentiment

def rmsle(y, y0) :
    #y = [x if x>0 else 0 for x in y]
    #y0 = [x if x>0 else 0 for x in y0]
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))

def main():
    df = pd.read_csv(traindata)
    df['status'] = pd.get_dummies(df['status'])
    df['homepage'] = pd.get_dummies(df['homepage'].notnull()) # apply((lambda x: x if x==pd.NaN else 1))
    temp_df = pd.get_dummies(df['original_language'])
    #print(temp_df.sum())
    temp_df = temp_df[['en', 'fr', 'hi', 'ja', 'ru']]
    df['spoken_languages'] = df['spoken_languages'].fillna(value="[{'iso_0': 'any', 'name': 'ANY'}]")
    temp_df['num_lang'] = df['spoken_languages'].apply(lambda z: len(json.loads(str(z).replace("'",'"'))))
    sent_file = open('AFINN-111.txt')
    scores = make_scores(sent_file)
    #df['production_companies'] = df['production_companies'].fillna(value="[{'iso_0': 'any', 'name': 'ANY'}]")
    #for i in range(3000) :
    #    print(reap.sub('', str(df['production_companies'].iloc[i])).replace("Donners' Company", 'Donners').replace("O'Connor Brothers","Connor Brothers").replace("d'Azur","dAzur").replace("l'Audiovisuel","lAudiovisuel").replace("d'Animation","dAnimation").replace("Mel's", "Mels").replace("Kids'", 'Kids').replace("\\xa0",'').strip())
    #    json.loads( reap.sub('', str(df['production_companies'].iloc[i])).replace("Donners' Company", 'Donners').replace("O'Connor Brothers","Connor Brothers").replace("d'Azur","dAzur").replace("Mel's", "Mels").replace("d'Animation","dAnimation").replace("l'Audiovisuel","lAudiovisuel").replace("\\xa0",'').replace('"Tor"','Tor').replace('"DIA"', 'DIA').replace("Kids'", 'Kids').replace('"Tsar"', 'Tsar').replace("Gettin'", "Gettin").replace("'",'"').strip() )
    
    #temp_df['num_comp'] = df['production_companies'].apply(lambda z: len(json.loads(str(z).replace("'",'"'))))
    df['production_countries'] = df['production_countries'].fillna(value="[{'iso_0': 'any', 'name': 'ANY'}]")
    temp_df['num_cont'] = df['production_countries'].apply(lambda z: len(json.loads(str(z).replace("'",'"'))))

    df['overview'] = df['overview'].fillna(value="")
    df['tagline'] = df['tagline'].fillna(value="")
    df['len_tagline'] = df['tagline'].apply(lambda x: len(x))
    df['belongs_to_collection'] = df['belongs_to_collection'].fillna(value='NO')
    df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda x: 0 if x=='NO' else 1)    
    df['words_in_tagline'] = df['tagline'].apply(lambda x: len(x.split(' ')))
    df['len_overview'] = df['overview'].apply(lambda x: len(x))
    df['words_in_overview'] = df['overview'].apply(lambda x: len(x.split(' ')))    
    df['len_title'] = df['title'].apply(lambda x: len(x))
    temp_df['neg_title'] = df['title'].apply(lambda x: neg_assess(x.strip(), scores))
    temp_df['pos_title'] = df['title'].apply(lambda x: pos_assess(x.strip(), scores))
    temp_df['neg_tagline'] = df['tagline'].apply(lambda x: neg_assess(x.strip(), scores))
    temp_df['pos_tagline'] = df['tagline'].apply(lambda x: pos_assess(x.strip(), scores))
    temp_df['neg_overview'] = df['overview'].apply(lambda x: neg_assess(x.strip(), scores))
    temp_df['pos_overview'] = df['overview'].apply(lambda x: pos_assess(x.strip(), scores))

    df = df.join(temp_df)
    df['words_in_title'] = df['title'].apply(lambda x: len(x.split(' ')))
    df[['release_month','release_day','release_year']]=df['release_date'].str.split('/',expand=True).replace(np.nan, 0).astype(int)
    df['release_year'] = df['release_year']
    df.loc[ (df['release_year'] <= 18) & (df['release_year'] < 100), "release_year"] += 2000
    df.loc[ (df['release_year'] > 18)  & (df['release_year'] < 100), "release_year"] += 1900
    
    print(df.columns)
    print(df.info())
    print(df.corr()['revenue'])
    #sns.heatmap(df.corr(), cmap='YlGnBu', annot=True, linewidths = 0.2);
    #plt.show()
    #sns.jointplot(df.budget, df.revenue);
    #sns.jointplot(df.popularity, df.revenue);
    #sns.jointplot(df.runtime, df.revenue);
    #plt.show()
    X = df[['budget', 'popularity', 'runtime', 'status', 'homepage', 'en', 'fr', 'hi', 'ja', 'ru', 'num_lang', 'num_cont', 'len_title', 'words_in_title', 'len_tagline', 'words_in_tagline', 'len_overview', 'words_in_overview', 'release_year', 'belongs_to_collection', 'neg_title', 'pos_title', 'neg_tagline', 'pos_tagline', 'neg_overview', 'pos_overview']]
    #X['budget'] = np.log1p(X['budget'])
    Y = np.log1p(df['revenue'])
    print(Y.min())
    fills = {'budget': 2.25e+7, 'popularity': 8.46, 'runtime': 107.85}
    X = X.fillna(value=fills)
    X['budget'] = np.log1p(X['budget'])
    #X = pd.DataFrame(preprocessing.normalize(X), columns = X.columns)

    average_revenue=Y.mean()
    median_revenue=Y.median()
    aver = np.ones(Y.size)*average_revenue
    print("Dummy evaluation = {}".format(sqrt(mean_squared_error(Y, aver))))
    aver2 = np.ones(Y.size)*median_revenue
    print("Dummy evaluation 2 = {}".format(sqrt(mean_squared_error(Y, aver2))))

    selt = X.columns
    XX = X[['budget', 'popularity', 'runtime', 'len_title', 'words_in_title', 'len_tagline', 'words_in_tagline', 'len_overview', 'words_in_overview', 'release_year', 'neg_overview', 'pos_overview']]
    model = xgb.XGBRegressor(max_depth=5, 
                     learning_rate=0.01, 
                     n_estimators=10000, 
                     objective='reg:linear', 
                     gamma=1.85, 
                     silent=True,
                     subsample=0.8, 
                     colsample_bytree=0.7, 
                     colsample_bylevel=0.5)
    cross = cross_val_score(model, XX, Y, cv=5, scoring='neg_mean_squared_error')
    print("XG Boost train evaluation = {0}".format(sqrt(-1*np.array(cross).mean())))
    model.fit(XX, Y)
    xgb.plot_importance(model)
    plt.show()

# https://github.com/saumiko/Movie-Revenue-Prediction/blob/master/Codes/SaveFeatures/Final.py
# https://www.kaggle.com/c/tmdb-box-office-prediction/data

if __name__ == '__main__':
    main()
