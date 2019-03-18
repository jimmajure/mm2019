'''
Created on Mar 5, 2019

@author: jim
'''
import matplotlib.pyplot as plt
from data import get_team_games
from data import get_ordinals

def display_stats():
    tg = get_team_games()
    tg = tg.sort_values(['season','day'])
    teams = ['Virginia','Kansas','Xavier']
    
    year = 2018
    pltnum=1
    
    for team in teams:
    
        games = tg[(tg['team'] == team) & (tg['season'] == year)]
        games = games.merge(tg[['season','teamid','day','off_rating','ast_ratio','def_rating']] \
                    .rename(columns={"teamid":"opp_teamid"}), on=['season','opp_teamid','day'], suffixes = ("","_opp"))
    
        plt.figure(pltnum)
        pltnum+=1
        
        plt.subplot(311)
    
        plt.plot(games['day'].values, games['off_rating'].values, label=team)
        plt.plot(games['day'].values, games['off_rating'].rolling(6,center=False).mean().values, label="mean")
        plt.plot(games['day'].values, games['off_rating'].rolling(6,center=False).median().values, label="median")
        plt.ylim(50,160)
        plt.grid(True)
        plt.legend()
    
        plt.subplot(312)
        plt.plot(games['day'].values, games['ast_ratio'].values, label=team)
        plt.plot(games['day'].values, games['ast_ratio'].rolling(6,center=False).mean().values, label="mean")
        plt.plot(games['day'].values, games['ast_ratio'].rolling(6,center=False).median().values, label="median")
        plt.grid(True)
        plt.ylim(0,50)
        plt.legend()
        
        plt.subplot(313)
        plt.plot(games['day'].values, games['def_rating'].values, label=team)
        plt.plot(games['day'].values, games['def_rating'].rolling(6,center=False).mean().values, label="mean")
        plt.plot(games['day'].values, games['def_rating'].rolling(6,center=False).median().values, label="median")
        plt.grid(True)
        plt.ylim(50,150)
        plt.legend()
        
    plt.show()

def display_ordinal():
    ordinal = get_ordinals()
    print("hi")

if __name__ == '__main__':
    display_ordinal()