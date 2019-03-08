'''
Created on Feb 17, 2019

@author: jim
'''
import numpy as np
import pandas as pd
from data import *
from power import get_power


def load_model():
    from joblib import load
    return load('../data/model.joblib')

def load_data():
    games = get_all_tourney_games()
    power = get_power()
    team_season = get_rolling_team_season()
    
    return games, power, team_season

def calc_score():
    tgp = get_played_tourney_games()
    predictions = predict()
    
    predictions = predictions[['season','teamid','opp_teamid','team','opp_team','prob']]
    
    tgp = tgp.merge(predictions,on=['season','teamid','opp_teamid'])
    
    losses = tgp['win']*np.log(tgp['prob']) + \
        (1-tgp['win'])*np.log((1-tgp['prob']))
    print("score: {}".format(-losses.sum()/len(losses)))

def predict():
    clf = load_model()
    games, power, team_season = load_data()
    
    fields = ['power']
    stats = ['off_rating', 'def_rating', 'to_ratio', 'true_shoot', 'eff_fg']
    games.reset_index()
    mgames = games.merge(power[['season','teamid','power']], on=['season','teamid'])
    mgames = mgames.merge(power[['season','teamid','power']], left_on=['season','opp_teamid'], 
                          right_on=['season','teamid'], suffixes=("","_opp"))
    
    team_season = team_season[['season','teamid']+stats]
    mgames = mgames.merge(team_season, on=['season','teamid'])
    mgames = mgames.merge(team_season.rename(columns={'teamid':"opp_teamid"}), 
                          on=['season','opp_teamid'], suffixes=("","_opp"))
    
    model_fields = stats + ['power']
    model_fields = model_fields + [f+"_opp" for f in model_fields]
    
    X = mgames[model_fields].values
    X_pred = clf.predict_proba(X)
    
    probs = pd.DataFrame(X_pred[:,1], columns=(['prob']))
    mgames = mgames.join(probs)
    with open("../data/predictions.csv",mode="w") as f:
        mgames.to_csv(f, na_rep='NULL', index=False)

    return mgames
    
if __name__ == '__main__':
    pd.set_option('display.width', 400)
    calc_score()
    pass