'''
Created on Feb 17, 2019

@author: jim
'''
import numpy as np
import pandas as pd
from data import *
from data import get_power
from sklearn_model_score import Scorer


def load_models():
    from joblib import load
    for m in ['./data/model.joblib','./data/model2.joblib']:
        yield load(m)

def load_data():
    games = get_all_tourney_games()
    power = get_power()
    team_season = get_rolling_team_season()
    
    return games, power, team_season

def calc_score():
    tgp = get_played_tourney_games()
    predictions, names = predict()
    
    predictions = predictions[['season','teamid','opp_teamid','team','opp_team']+names]
    
    tgp = tgp.merge(predictions,on=['season','teamid','opp_teamid'])
    
    for n in names:
        losses = tgp['win']*np.log(tgp[n]) + \
            (1-tgp['win'])*np.log((1-tgp[n]))
        print("{}: {}".format(n, -losses.sum()/len(losses)))

def predict():
    games, power, team_season = load_data()
    games.reset_index()

    print(team_season.groupby('season').groups.keys())
    games = games.merge(team_season, on=['season','teamid'])
    games = games.merge(team_season.rename(columns={'teamid':"opp_teamid"}), 
                        on=['season','opp_teamid'], suffixes=("","_opp"))

    games = games.copy().merge(power[['season','teamid','power']], on=['season','teamid'])
    games = games.merge(power[['season','teamid','power']], left_on=['season','opp_teamid'], 
                        right_on=['season','teamid'], suffixes=("","_opp"))

    mgames = games.copy()[['season','teamid','team','opp_teamid','opp_team']]

    names = []
    for m in load_models():
        fields = m['fields']
        fields = fields + [f+"_opp" for f in fields]
        clf = m['clf']
        name = m['name']
        names.append(name)
                
        X = games[fields].values
        X_pred = clf.predict_proba(X)
        
        probs = pd.DataFrame(X_pred[:,1], columns=([name]))
        mgames = mgames.join(probs)

    with open("./data/predictions.csv",mode="w") as f:
        mgames.to_csv(f, na_rep='NULL', index=False)

    return mgames, names
    
if __name__ == '__main__':
    pd.set_option('display.width', 400)
    calc_score()
    pass