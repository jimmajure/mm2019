
'''
'''

from __future__ import print_function

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from data import *
from data import get_power
from scipy.stats import norm

import pandas as pd
import numpy as np
from sklearn.preprocessing._encoders import OneHotEncoder
from itertools import chain, combinations
from copy import copy

class Scorer:
    def __init__(self,clf, stddev):
        self.clf = clf
        self.stddev = stddev

    def predict_proba(self, X):
        pred = self.clf.predict(X)

        lose = norm.cdf(0,loc=pred, scale=self.stddev)
        win = 1-lose

        return np.array([lose,win]).T

    def predict(self, X):
        return self.clf.predict(X)

def load_base_data():
    fields = ['off_rating','def_rating','ast_ratio','to_ratio','true_shoot','eff_fg','power']

    team_games = get_team_games()
    rolling_team_games = get_rolling_team_games()
    power = get_power()[['season','teamid','power']]
    
    team_games = team_games[['season','score','win','gamedate','teamid','team','opp_teamid']]
    rolling_team_games = rolling_team_games[['season','teamid','gamedate']+fields[:-1]]
    team_games = team_games.merge(rolling_team_games, on=['season','teamid','gamedate'])

    games = team_games.copy()
    games = games[games['teamid'] < games['opp_teamid']]

    opp_games = team_games.drop(['opp_teamid'], axis="columns")\
        .rename(columns={"teamid":"opp_teamid"})

    games = games.merge(opp_games, on=['season','opp_teamid','gamedate'], suffixes=("","_opp"))
    
    games = games.merge(power, on=['season','teamid'])
    games = games.merge(power.rename(columns={'teamid':"opp_teamid"}), on=['season','opp_teamid'], suffixes=("","_opp"))

    return games

def load_data_score():
    
    games = load_base_data()

    fields = ['off_rating','def_rating','ast_ratio','to_ratio','true_shoot','eff_fg','power']

    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))
    
    super_fields = powerset(fields)

    for fields in super_fields:
        all_fields = list(fields)

        all_fields = all_fields + [f + "_opp" for f in all_fields]
        X = games[all_fields].values

        y = games[['win','score','score_opp']].copy()
        y['score_diff'] = y['score']-y['score_opp']
        y.drop(['score','score_opp'], axis='columns')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1)
    
        yield (X_train, y_train), (X_test, y_test), all_fields

def persist_model():
    fields = ['off_rating', 'def_rating','ast_ratio', 'to_ratio', 'true_shoot', 'power']
    fields_all = fields + [f+"_opp" for f in fields]
    games = load_base_data()

    X = games[fields_all].values

    y = games[['win','score','score_opp']].copy()
    y['score_diff'] = y['score']-y['score_opp']
    y.drop(['score','score_opp'], axis='columns')

    regr = Ridge(alpha=1)
    regr.fit(X, y['score_diff'])        

    y_pred = regr.predict(X)

    stddev = np.sqrt(np.sum((y_pred-y['score_diff'])**2)*1/(len(y_pred)-2))
    print(stddev)
    scr = Scorer(regr, stddev)

    model = { \
        "fields":fields,
        "clf": scr,
        "name": "model2"
        }



    
    from joblib import dump
    dump(model, "./data/model2.joblib")    
    
def eval(win, score_diff):
    def msd(sd):
        if sd < 0:
            return 0
        else:
            return 1
    
    results = [1-abs(w-msd(d)) for w,d in zip(win, score_diff)]
    
    return sum(results)/len(results)

def train_model_score():
    results = []
    for (x_train, y_train), (x_test, y_test), fields in load_data_score():

        regr = Ridge(alpha=1)
        regr.fit(x_train, y_train['score_diff'])        

        y_test_pred = regr.predict(x_test)

        stddev = np.std((y_test_pred-y_test['score_diff'])**2)
        print(1-norm.cdf(0,loc=y_test_pred, scale=stddev*(1+1/len(y_test))))

        results.append([r2_score(y_test['score_diff'], y_test_pred), eval(y_test['win'], y_test_pred),fields])
    
    results = sorted(results, key=lambda x: x[1], reverse=True)
    for r in results:
        print("{:5.4f}  {:5.4f}  {}".format(*r))
        
    

if __name__ == '__main__':
    # train_model_score()
    persist_model()