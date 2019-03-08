
'''
'''

from __future__ import print_function

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from data import *
from power import get_power

import pandas as pd
import numpy as np
from sklearn.preprocessing._encoders import OneHotEncoder
from itertools import chain, combinations
from copy import copy



def load_data():
     
    fields = ['off_rating', 'def_rating', 'to_ratio', 'true_shoot', 'eff_fg']
    
    team_games = get_rolling_team_games()
    power = get_power()
    team_season = get_team_season()
    coaches = get_coaches()
    
#    remove redundant games
    games = team_games.copy()
    games = games[games['teamid']<games['opp_teamid']]
    games = games[['season','win','gamedate','teamid','opp_teamid']+fields]
    num_games = games.shape[0]
    
    opp_games = team_games[['season','win','gamedate','teamid']+fields] \
        .rename(columns={"teamid":"opp_teamid"})

    games = games.merge(opp_games, on=['season','opp_teamid','gamedate'], suffixes=("","_opp"))
    num_games = games.shape[0]
#     if num_games != games.shape[0]:
#         raise Exception('num games incorrect. Expected {}, got {}'.format(num_games, games.shape[0]))

    
    games = games.merge(power, on=['season','teamid'])
    power_opp = power.rename(columns={"teamid":"opp_teamid"})
    games = games.merge(power_opp, on=['season','opp_teamid'], suffixes=("","_opp"))
    if num_games != games.shape[0]:
        raise Exception('num games incorrect. Expected {}, got {}'.format(num_games, games.shape[0]))
    
#     team_season = team_season[['season','teamid']+fields]
#     games = games.merge(team_season, on=['season','teamid'])
#     games = games.merge(team_season.rename(columns={"teamid":"opp_teamid"}),
#                         on=['season','opp_teamid'], suffixes=("","_opp"))
#     if num_games != games.shape[0]:
#         raise Exception('num games incorrect')
    
#     games = games.merge(coaches, on=['season','teamid', 'day'])
#     games = games.merge(coaches.rename(columns={"teamid":"opp_teamid"}),
#                         on=['season','opp_teamid','day'], suffixes=("","_opp"))
#     if num_games != games.shape[0]:
#         raise Exception('num games incorrect: was {}, expected {}'.format(games.shape[0], num_games))
    
    fields = fields + ['power']
    fields = fields + [f+'_opp' for f in fields]
#     column_trans = ColumnTransformer(
#      [('coach', OneHotEncoder(),['coach']),
#       ('coach_opp', OneHotEncoder(),['coach_opp'])],
#      remainder='passthrough')
    
#     X = column_trans.fit_transform(games[fields])
    X = games[fields].values
    y = games['win'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15)

    return (X_train, y_train), (X_test, y_test)

def load_data2():
    
    fields = ['off_rating','def_rating','ast_ratio','to_ratio','true_shoot','eff_fg','power']
    
    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))
    
    team_games = get_rolling_team_games()
    power = get_power()[['season','teamid','power']]

    super_fields = powerset(fields)
    
    for fields in super_fields:
        all_fields = list(fields)
        some_fields = copy(all_fields)
        if 'power' in some_fields:
            some_fields.remove('power')
        
        games = team_games.copy()
        games = games[games['teamid'] < games['opp_teamid']]
        games = games[['season','win','gamedate','teamid','opp_teamid']+some_fields]
        
        opp_games = team_games[['season','win','gamedate','teamid']+some_fields] \
            .rename(columns={"teamid":"opp_teamid"})
    
        games = games.merge(opp_games, on=['season','opp_teamid','gamedate'], suffixes=("","_opp"))
        
        games = games.merge(power, on=['season','teamid'])
        games = games.merge(power.rename(columns={'teamid':"opp_teamid"}), on=['season','opp_teamid'], suffixes=("","_opp"))
        
    #     games = games[games['season']==2018]
        
        all_fields = all_fields + [f + "_opp" for f in all_fields]
        X = games[all_fields].values
    
        y = games['win'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1)
    
        yield (X_train, y_train), (X_test, y_test), all_fields
# 0.660447  ['off_rating', 'def_rating', 'ast_ratio', 'to_ratio', 'true_shoot', 'off_rating_opp', 'def_rating_opp', 'ast_ratio_opp', 'to_ratio_opp', 'true_shoot_opp']

def get_classifiers():
    clf = LogisticRegression(solver="liblinear", max_iter=4000)

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        'solver': ['liblinear'],
        'C': [0.7, 0.8, 1.0],
    }
    yield "logistic regression", clf, param_grid
    
#     clf = SVC()
#     param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [100, 1000]}]
#     ,
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#     yield "SVM", clf, param_grid
     
#     clf = GaussianNB()
#     param_grid = [{"var_smoothing" : [1e-9]}]
#     yield "Gaussian Naieve Bayes", clf, param_grid
    return

def train_model2():
    results = []
    for (x_train, y_train), (x_test, y_test), fields in load_data2():
        for name, clf, grid in get_classifiers():
        
            scores = ['accuracy']
        
            for score in scores:
        
                clf = GridSearchCV(clf, grid, cv=5,
                                   scoring='{}'.format(score),
                                   n_jobs=-1)
                clf.fit(x_train, y_train)

                results.append((clf.best_score_,fields)) 
    
    results = sorted(results, key=lambda x: x[0], reverse=True)
    for r in results:
        print("{:5f}  {}".format(r[0], r[1]))
        

def train_model():
    (x_train, y_train), (x_test, y_test) = load_data()
        
    for name, clf, grid in get_classifiers():
        
        scores = ['accuracy']
    
        for score in scores:
    
            print("# Tuning hyper-parameters for {}/{}".format(name,score))
            print()
        
            clf = GridSearchCV(clf, grid, cv=5,
                               scoring='{}'.format(score),
                               n_jobs=-1)
            clf.fit(x_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()
        
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(x_test)
            print(classification_report(y_true, y_pred))
            print()

def persist_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    clf = LogisticRegression(solver="liblinear", max_iter=10000, C=1.0)
    clf.fit(x_train, y_train)
    
    from joblib import dump
    dump(clf, "../data/model.joblib")
    
    

if __name__ == '__main__':
#     train_model2()
#     train_model()
    persist_model()
    
    