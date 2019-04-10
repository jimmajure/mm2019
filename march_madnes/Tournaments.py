'''
A package to deal with NCAA basketball tournaments based on the
file formats distributed in the kaggle tournament challenges.

Created on Mar 1, 2018

@author: jim
'''

import pandas as pd
from pandas.core.frame import DataFrame
import random
from math import log

def r_value(p_value):
    return random.random()

def one(p_value):
    if (p_value>0.5):
        return 0.0
    else:
        return 1.0

class Seed(object):
    
    def __init__(self, team: int, season: int, seed: str, teams: DataFrame):
        self.team = team
        self.seed = seed
        self.season = season
        self.team_name = teams['team'][teams['teamid']==team].iloc[0]
        self.simulation_winner = self
        self.actual_winner = self
        
    def simulate(self, p_a_beats_b, r_value, scorer, resulter):
        '''
        For a Seed, the seed itself is the winner, the probability is 1.0,
        the cumulative simulated and actual scores are 0.0, and there are no results to report.
        '''
        return self, 0, 1.0, 0

    def results(self):
        return []

class Slot(Seed):
    
    def __init__(self, slot: str, season: int, slot1: Seed, slot2: Seed):
        self.season = season
        self.slot = slot
        self.slot1 = slot1
        self.slot2 = slot2
    
    def simulate(self, p_a_beats_b, r_value, scorer, resulter):
        '''
        Simulate the playing of this slot.
        '''
        team1, score1, pvalue1, actual_score1 = self.slot1.simulate(p_a_beats_b, r_value, scorer, resulter)
        team2, score2, pvalue2, actual_score2 = self.slot2.simulate(p_a_beats_b, r_value, scorer, resulter)

        if team1.team > team2.team:
            tmp = team1
            team1 = team2
            team2 = tmp

        p_value = p_a_beats_b(self.season, team1, team2)
        
        # team1 wins with probability p_value...
        r_v = r_value(p_value)
        if r_v <= p_value:
            self.simulation_winner = team1
            cum_pvalue = p_value * pvalue1 * pvalue2
        else:
            self.simulation_winner = team2
            cum_pvalue = (1-p_value) * pvalue2 * pvalue1

        actual_score = None
        if resulter:
            actual_winner = resulter(self.season, team1, team2)
            actual_score = scorer(actual_winner, self) + \
                actual_score1 + actual_score2
        
        score = scorer(self.simulation_winner, self)

        return self.simulation_winner, score+score1+score2, cum_pvalue, actual_score

    def results(self):
        result = self.slot1.results() + self.slot2.results()
        result.append("Slot: {}: {} vs {} => {}"\
            .format(self.slot, self.slot1.simulation_winner.team_name, \
                self.slot2.simulation_winner.team_name,self.simulation_winner.team_name))
        return result

def default_scorer(winner, slot):
    '''
    The default is that each win is worth 1 point.
    '''
    return 1

def round_scorer(winner: Seed, slot: Slot):
    '''
    Scoring based on the rules of round points * seed.
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
    
    if not winner:
        result = 0
    else:
        if winner.seed[-1:] in ['a','b']:
            seed = 16
        else:
            seed = int(winner.seed[-2:])
    
        result = result * seed

    return result

        
class Tournament(object):
    '''
    Represents an individual NCAA tournament. Builds the tournament 
    '''

    def __init__(self, season: int, slots: DataFrame, seeds: DataFrame, teams: DataFrame):
        '''
        Read each seed and slot.
        '''
        seeds_dict = {}
        season_seeds = seeds[seeds['season']==season]
        for _, row in season_seeds.iterrows():
            seeds_dict[row['seed']] = Seed(row['teamid'], season, row['seed'], teams)
        
        season_slots = slots[slots['season']==season]
        slots_dict = {}
        
        for _, row in season_slots.iterrows():
            # first get all the seeds...
            strongseed = seeds_dict.get(row['strongseed'])
            weakseed = seeds_dict.get(row['weakseed'])
            
            if strongseed and weakseed:
                slots_dict[row['slot']] = Slot(row['slot'], season, strongseed, weakseed)
        
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

                slots_dict[row['slot']] = Slot(row['slot'], season, strongseed, weakseed)
        
        self.final_slot = slots_dict['R6CH']
        
    def simulate(self, p_a_beats_b, r_value=r_value, scorer=default_scorer, resulter=None):
        winner, score, pvalue, actual_score = self.final_slot.simulate(p_a_beats_b, r_value, scorer, resulter)
        return winner, score, pvalue, actual_score

    def results(self):
        return self.final_slot.results()
    
        
def load_data():
    seasons = pd.read_csv('./data/Seasons.csv')
    teams = pd.read_csv("./data/Teams.csv")
    seeds = pd.read_csv("./data/NCAATourneySeeds.csv")
    slots = pd.read_csv("./data/NCAATourneySlots.csv")

    return teams, seasons, slots, seeds


def load_tournaments():
    tournaments = {}
    
    teams, seasons, slots, seeds = load_data()
    
    for season in seasons['season']:
        tournaments[season] = Tournament(season, slots, seeds, teams)
    
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

