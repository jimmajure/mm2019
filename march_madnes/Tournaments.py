'''
A package to deal with NCAA basketball tournaments based on the
file formats distributed in the kagle tournament challenges.

Created on Mar 1, 2018

@author: jim
'''

import pandas as pd
from pandas.core.frame import DataFrame
import random

def r_value(p_value):
    return random.random()

def one(p_value):
    if (p_value>0.5):
        return 0.0
    else:
        return 1.0
    
class NCAA_PFunction(object):
    
    def __init__(self, winlose_season: DataFrame, clf, vectorizer):
        self.winlose_season = winlose_season
        self.clf = clf
        self.vectorizer = vectorizer
        
    def get_p_function(self):
        def p_a_beats_b(a: Seed, b:Seed):
            
            a_data = self.winlose_season[(self.winlose_season['season']==a.season) & (self.winlose_season['team']==a.team)]
            b_data = self.winlose_season[(self.winlose_season['season']==b.season) & (self.winlose_season['team']==b.team)]
            
            data = DataFrame({
                'wgt_winpct_diff':[a_data['wgt_winpct'].iloc[0] - b_data['wgt_winpct'].iloc[0]],
                'oscore_dscore_diff':[a_data['score_med'].iloc[0] - b_data['opp_score_med'].iloc[0]],
                'dscore_oscore_diff':[a_data['opp_score_med'].iloc[0] - b_data['score_med'].iloc[0]],
                'conf_id':[a_data['conf_class'].iloc[0] + b_data['conf_class'].iloc[0]],
                'ast_to':[a_data['ast_to_med'].iloc[0]],
                'fg_pct_diff':[a_data['fg_pct_med'].iloc[0]-b_data['opp_fg_pct_med'].iloc[0]],
                'oor_dor_diff':[a_data['or_med'].iloc[0]-b_data['opp_or_med'].iloc[0]],          
                'dor_oor_diff':[a_data['opp_or_med'].iloc[0]-b_data['or_med'].iloc[0]]       
            })
            
            X = data[['wgt_winpct_diff', 'oscore_dscore_diff', 'dscore_oscore_diff', 'conf_id', 'ast_to','fg_pct_diff', 'oor_dor_diff', 'dor_oor_diff']]
#             X = data[['wgt_winpct_diff', 'conf_id']]
#             print(X)
#             print(self.clf.predict_proba(self.vectorizer.transform(X)))
            result = self.clf.predict_proba(self.vectorizer.transform(X))[0][1]
            
            return result
            
        return p_a_beats_b
    

class Seed(object):
    
    def __init__(self, team: int, season: int, seed: str, teams:DataFrame):
        self.team = team
        self.seed = seed
        self.season = season
        self.team_name = teams['name'][teams['team']==team].iloc[0]

        self.actual_winner = self
        
    def simulate(self, p_a_beats_b, r_value, scorer):
        '''
        For a Seed, the seed itself is the winner, the probability is 1.0,
        the cumulative simulated and actual scores are 0.0, and there are no results to report.
        '''
        return self, 1.0, 0.0, 0.0, None
    
    def set_winners(self, results: DataFrame):
        '''
        Sets the actual winner. This is a noop for a Seed.
        '''
        #noop
        pass

class Slot(Seed):
    
    def __init__(self, slot: str, slot1: Seed, slot2: Seed):
        self.slot = slot
        self.slot1 = slot1
        self.slot2 = slot2
    
    def simulate(self, p_a_beats_b, r_value, scorer):
        '''
        Simulate the playing of this slot.
        '''
        result = []
        team1, t1p, t1score, t1_acc_score, result1 = self.slot1.simulate(p_a_beats_b, r_value, scorer)
        if result1:
            result += result1

        team2, t2p, t2score, t2_acc_score, result2 = self.slot2.simulate(p_a_beats_b, r_value, scorer)
        if result2:
            result += result2
            
        p_function = p_a_beats_b.get_p_function()
        p_value = p_function(team1, team2)
        
        
        # team1 wins with probability p_value...
        cum_actual_score = t1_acc_score + t2_acc_score
        cum_score = t1score + t2score
        r_value = r_value(p_value)
        if r_value <= p_value:
            winner = team1
            cum_p = t1p * p_value
            cum_score += (p_value * scorer(winner, self))
        else:
            winner = team2
            cum_p = t2p * (1-p_value)
            cum_score += ((1-p_value) * scorer(winner, self))

        if self.actual_winner.team == winner.team:
            cum_actual_score += scorer(winner, self)

        result.append({"slot":self.slot, "winner": winner,
                "cumulative_p": cum_p, "cumulative_score": cum_score,
                "team1": team1, "team2":team2,
                "p_value":p_value, "r_value":r_value,
                "actual_winner": self.actual_winner,
                "cumulative_actual_score": cum_actual_score})
        
        return winner, cum_p, cum_score, cum_actual_score, result
    
    def set_winners(self, results: DataFrame):
        self.slot1.set_winners(results)
        self.slot2.set_winners(results)
        
        team1 = self.slot1.actual_winner
        team2 = self.slot2.actual_winner
        
        # lookup the winner of the slot
        if results[(results['wteam']==team1.team)&(results['lteam']==team2.team)].shape[0]>0:
            self.actual_winner = team1
        elif results[(results['wteam']==team2.team)&(results['lteam']==team1.team)].shape[0]>0:
            self.actual_winner = team2
        else:
            raise ValueError("Can't find winner for slot {}".format(self.slot))

def default_scorer(winner, slot):
    '''
    The default scoring function for a given seed winning
    a given slot with a given probability.
    
    The default is that each win is worth 1 point.
    '''
    return 1

def round_scorer(winner: Seed, slot: Slot):
    '''
    The default scoring function for a given seed winning
    a given slot with a given probability.
    
    .
    '''
    result = 0
    if slot.slot[0:2] == "R1":
        result = 1
    elif slot.slot[0:2] == 'R2':
        result = 2   
    elif slot.slot[0:2] == 'R3':
        result = 4
    elif slot.slot[0:2] == 'R4':
        result = 8
    elif slot.slot[0:2] == 'R5':
        result = 16
    elif slot.slot[0:2] == 'R6':
        result = 32
    elif slot.slot[0] in ['W','X','Y','Z']:
        result = 0
    else:
        raise ValueError("Can't classify slot: {}".format(slot.slot))
    
    if winner.seed[-1:] in ['a','b']:
        seed = 16
    else:
        seed = int(winner.seed[-2:])
    
    return result * seed

        
class Tournament(object):
    '''
    Represents an individual NCAA tournament.
    '''

    def __init__(self, season: int, slots: DataFrame, seeds: DataFrame, teams: DataFrame, results: DataFrame):
        '''
        Read each seed and slot.
        '''
        seeds_dict = {}
        season_seeds = seeds[seeds['season']==season]
        for _, row in season_seeds.iterrows():
            seeds_dict[row['seed']] = Seed(row['team'], season, row['seed'], teams)
        
        season_slots = slots[slots['season']==season]
        slots_dict = {}
        
        for _, row in season_slots.iterrows():
            # first get all the seeds...
            strongseed = seeds_dict.get(row['strongseed'])
            weakseed = seeds_dict.get(row['weakseed'])
            
            if strongseed and weakseed:
                slots_dict[row['slot']] = Slot(row['slot'], strongseed, weakseed)
        
        for _, row in season_slots.iterrows():
            if not row['slot'] in slots_dict:
                strongseed = seeds_dict.get(row['strongseed'])
                if not strongseed:
                    strongseed = slots_dict[row['strongseed']]
                weakseed = seeds_dict.get(row['weakseed'])
                if not weakseed:
                    weakseed = slots_dict[row['weakseed']]
                    
                if not (strongseed and weakseed):
                    raise ValueError("Problem loading slot {}, {}/{}".format(row['slot'],row['strongseed'],row['weakseed']))

                slots_dict[row['slot']] = Slot(row['slot'], strongseed, weakseed)
        
        self.final_slot = slots_dict['R6CH']
        self.set_winners(results[results['season']==season])
        
    def simulate(self, p_a_beats_b, r_value=r_value, scorer=default_scorer):
        winner, cum_p, cum_score, cum_actual_score, result = self.final_slot.simulate(p_a_beats_b, r_value, scorer)
        return winner, cum_p, cum_score, cum_actual_score, result
    
    def set_winners(self, results:DataFrame):
        self.final_slot.set_winners(results)
        
def load_data():
    seasons = pd.read_csv('Seasons.csv')
    teams = pd.read_csv("Teams.csv")
    seeds = pd.read_csv("NCAATourneySeeds.csv")
    slots = pd.read_csv("NCAATourneySlots.csv")
    results = pd.read_csv("NCAATourneyCompactResults.csv")

    regdtl = pd.read_csv("RegularSeasonDetailedResults.csv")
    teams = pd.read_csv("Teams.csv")
    teams = teams.rename(columns={"team_id":"team","team_name":"name"})
    team_conf = pd.read_csv("TeamConferences.csv")
    confs = pd.read_csv("Conferences.csv")
    team_conf = team_conf.merge(confs[['conf_id','conf_class']],on=('conf_id'))

    return regdtl, teams, team_conf, seasons, slots, seeds, results


def load_tournaments():
    tournaments = {}
    
    regdtl, teams, team_conf, seasons, slots, seeds, results = load_data()
    
    for season in seasons['season']:
        tournaments[season] = Tournament(season, slots, seeds, teams, results)
    
    return tournaments
    
def display_simulation(winner, simulation_results, round_prefx=None):
    '''
    Display a list of simulation results...
    '''
    def sorter(x):
        if len(x['slot']) == 3:
            return "A"+x['slot']
        else:
            return x['slot']

    sim_srt = sorted(simulation_results, key=sorter, reverse=True)
    print("Winner: {}".format(winner.team_name))
    for r in sim_srt:
        do_print = True
        if round_prefx:
            do_print = r['slot'][:2] in round_prefx
        if do_print:
            if r['winner'] == r['team1']:
                team1 = '*{} ({})*'.format(r['team1'].team_name, r['team1'].seed)
                team2 = '{} ({})'.format(r['team2'].team_name, r['team2'].seed)
            else:
                team1 = '{} ({})'.format(r['team1'].team_name, r['team1'].seed)
                team2 = '*{} ({})*'.format(r['team2'].team_name, r['team2'].seed)
            if r['winner'] == r['actual_winner']:
                actual_winner = '*{}*'.format(r['actual_winner'].team_name)
            else:
                actual_winner = r['actual_winner'].team_name
                
            print("{:>4}: {:>25} : {:<25} {:4.3f}/{:4.3f}; {:7.3f};    {:<20}; {:7.3f}".format(r['slot'],
                team1, team2,
                r['p_value'], r['r_value'], r['cumulative_p'],
                actual_winner, r['cumulative_actual_score']
            ))

