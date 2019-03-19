'''
Created on Mar 1, 2018

@author: jim
'''

from ncaa_tournament.Tournaments import load_tournaments, default_scorer,\
    display_simulation, NCAA_PFunction, round_scorer, r_value, one
import pickle

if __name__ == '__main__':
    
    with open('ncaa_learn.pickle', 'rb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        p_function = pickle.load(f)

    tournaments = load_tournaments()

    overalls = []
    max_cum = -1.0
    max_score = 0.0
    max_results = None
    max_winner = None
    for i in range(5):
        if i%50==0:
            print("{}: max: {}...".format(i, max_cum))
        winner, cum_p, cum_score, cum_actual_score, results = tournaments[2017].simulate(p_function,scorer=round_scorer,r_value=one)
#         display_simulation(winner, results, ['R6','R5','R4','R3'])
#         if cum_score > max_score:
#             max_score = cum_score
        if cum_p > max_cum:
            max_cum = cum_p
            max_results = results
            max_winner = winner
        overalls += results
    display_simulation(max_winner, max_results, ['R6','R5','R4','R3','R2','R1'])
