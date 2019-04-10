'''
Created on Mar 1, 2018

@author: jim
'''

from Tournaments import load_tournaments, default_scorer,\
    display_simulation, round_scorer, r_value, one

import pandas as pd
import pickle

if __name__ == '__main__':


    predictions = pd.read_csv("./data/predictions.csv")
    tourney_results = pd.read_csv("./data/NCAATourneyCompactResults.csv")
    
    def p_function(season, teama, teamb):
        p_value = predictions[(predictions.season == season)&(predictions.teamid == teama.team)& \
            (predictions.opp_teamid == teamb.team)].model2.values[0]

        if not p_value:
            raise Exception("Unable to find matchup for season {} for teams {} ({}) and {} ({})"\
                .format(season, teama.team_name, teama.team, teamb.team_name, teamb.team))
        return p_value

    def resulter(season, teama, teamb):
        tr = tourney_results
        result = tourney_results[((tr.season==season)&(((tr.wteamid==teama.team)&(tr.lteamid==teamb.team)) |\
            ((tr.wteamid==teamb.team)&(tr.lteamid==teama.team))))]
        
        if len(result) == 0:
            return None
        elif result.wteamid.values[0] == teama.team:
            return teama
        else:
            return teamb

    tournaments = load_tournaments()

    overalls = []
    tournament = tournaments[2018]
    for i in range(5000):
        winner, score, p_value, actual_score = \
            tournament.simulate(p_function,scorer=round_scorer,r_value=r_value, resulter=resulter)
        if actual_score:
            actual_score = int(actual_score)
        overalls.append((winner.team_name, int(score), p_value*1e14, tournament.results(), \
            actual_score))

    df = pd.DataFrame([[t,s,p,s2] for t,s,p,_,s2 in overalls], columns=['winner','sim_score','prob',\
        'act_score'])

    with open("./data/simulated.csv",mode="w") as f:
        df.to_csv(f, na_rep='NULL', index=False)


    results_sorted = sorted(overalls, key=lambda x:x[2], reverse=True)
    # for r in results_sorted[:50]:
    #     print("Winner: {:25s} Score: {:5d}; Prob: {:7f}".format(r[0],r[1],r[2]))

    for r in results_sorted[:20]:
        print("Winner: {:25s} Sim Score: {:5d}; Prob: {:7.5f}; Actual score: {:5d}".format(r[0],r[1],r[2],r[4]))
        for s in r[3]:
            print(s)