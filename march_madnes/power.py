'''
Created on Feb 17, 2019

@author: jim
'''
import numpy as np
import pandas as pd
from data import get_team_games
from data import load_data
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model.logistic import LogisticRegression


# generate a power score for each season
def get_power():
    result = []
    team_games = get_team_games()
    _, teams, _, _, _ = load_data()
    for season in range(2003,2019):
        
        tg = team_games[team_games['season']==season]
        X = tg[['teamid',"opp_teamid"]]
        y = tg['win'].values
        
        enc = OneHotEncoder(categories='auto')
        
        lr = LogisticRegression(solver="lbfgs")
        
        clf = Pipeline(steps=[
                ('encoder', enc),
                ('classifier', lr)
            ])    
        lr.fit(enc.fit_transform(X), y)
        
        pnames = [int(n[3:7]) for n in enc.get_feature_names()]
        name_score = list(zip([season]*len(pnames), pnames, lr.coef_[0]))
        # we only need the first half because the other half is the negative
        name_score = name_score[0:len(name_score)//2]
        pvalues = pd.DataFrame(name_score, columns=('season','teamid','power'))
        pvalues = pvalues.merge(teams[['teamid','team']])
        
        result += [pvalues]

    result = pd.concat(result, ignore_index=True)
    return result
        
