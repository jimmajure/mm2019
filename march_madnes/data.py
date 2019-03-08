'''
Created on Feb 17, 2019

@author: jim
'''
import numpy as np
import pandas as pd

base_stat_names = ['score', 
        'fgm', 'fga', 'fgm3', 'fga3', 'ftm', 'fta', 'or', 'dr',
        'ast', 'to', 'stl', 'blk', 'pf'
    ]

def get_teams():
    return pd.read_csv("../data/Teams.csv")

def get_seeds():
    return pd.read_csv("../data/NCAATourneySeeds.csv")

def get_coaches():
    return pd.read_csv("../data/TeamCoachesPerDay.csv")

def load_data():
    regdtl = pd.read_csv("../data/RegularSeasonDetailedResults.csv")
    teams = get_teams()
    coaches = get_coaches()
    conferences = pd.read_csv("../data/TeamConferences.csv")
    seeds = get_seeds()

    return regdtl, teams, coaches, conferences, seeds

# Figure out the tourney games that have been played so 
# the model can be scored...
def get_played_tourney_games():
    print("get_played_tourney_games...")

    with open("../data/NCAATourneyCompactResults.csv") as f:
        played_games = pd.read_csv(f)
        
    played_games = played_games[played_games['season']>2013]
    
    winners = pd.DataFrame({"season": played_games['season'],
                            "win":1,
                            "teamid":played_games["wteamid"],
                            "opp_teamid":played_games["lteamid"]})
    losers = pd.DataFrame({"season": played_games['season'],
                            "win":0,
                            "teamid":played_games["lteamid"],
                            "opp_teamid":played_games["wteamid"]})
    
    played_games = pd.concat([winners,losers], ignore_index=True)
    played_games = played_games[played_games['teamid']<played_games['opp_teamid']]
    
    print("get_played_tourney_games...done")
    return played_games
    
# Find all possible tourney matchups for probablity calculation...
def get_all_tourney_games():
    _,teams,_,_,seeds = load_data()
        
    from itertools import combinations
    def game_maker(season):
        out = [(t1,t2) for t1,t2 in combinations(sorted(list(season['teamid'])),2)]
        out = pd.DataFrame(out, columns=('teamid','opp_teamid'))
        return out
    
    games = seeds[seeds['season']>2013].groupby('season').apply(game_maker)
    games.reset_index(level=0,inplace=True)
    games = games.merge(teams[['teamid','team']])
    games = games.merge(teams[['teamid','team']].rename(columns={"teamid":"opp_teamid","team":"opp_team"}))
    
    return games

# create a file with a line for each game played by each team.
# This will include each game twice, once for each participant.
def get_team_games():
    print("get_team_games...")
    regdtl,teams, coaches, conferences, seeds = load_data()
    
    win_lose_names = ['teamid']+base_stat_names
    
    # show the games from the winner's perspective
    win_game = pd.DataFrame({"season":regdtl['season'],"win":1,
                            "loc":regdtl['wloc'],
                            "day":regdtl['daynum']})
    
    for wln in win_lose_names:
        win_game = win_game.join(
            pd.DataFrame({wln:regdtl["w"+wln], "opp_"+wln:regdtl['l'+wln]})
        )                        
    
    def home_away(ha):
        result = ha
        if ha=="H":
            result = 'A'
        elif ha == 'A':
            result = 'H'
        return result
    
    # show the games from the losers's perspective
    lose_game = pd.DataFrame({"season":regdtl['season'], "win": 0,
                            "loc":regdtl['wloc'].apply(home_away),
                            "day":regdtl['daynum']})
    
    for wln in win_lose_names:
        lose_game = lose_game.join(
            pd.DataFrame({wln:regdtl["l"+wln], "opp_"+wln:regdtl['w'+wln]})    
        )
    
    # concat the winners and losers to give a DF with 2 entries for each game...
    team_games = pd.concat([win_game, lose_game],ignore_index=True)
    # add the team names, conference names, coaches and power rating
    team_games = team_games.merge(teams[['teamid','team']], left_on=['teamid'], right_on=['teamid'])
    team_games = team_games.merge(coaches)
    team_games = team_games.merge(conferences)
#     team_games = team_games.merge(power)
    
    # add the date of the game for time-series...
    seasons = pd.read_csv("../data/Seasons.csv")
    team_games = team_games.merge(seasons[['season','dayzero']])
    team_games['gamedate'] = pd.to_datetime(team_games['dayzero'], format='%m/%d/%Y') \
        + pd.to_timedelta(team_games['day'], unit='d')

    # calcualate advanced stats for each game
    team_games['possessions'] = team_games['fga'] + 0.475*team_games['fta'] + \
        team_games['to'] - team_games['or']

    team_games['off_rating'] = 100 * (team_games['score']/team_games['possessions'])

    team_games['def_rating'] = 100 * (team_games['opp_score']/team_games['possessions'])

    team_games['ast_ratio'] = 100 * team_games['ast'] / (team_games['fga'] + 0.475*team_games['fta'] + \
        team_games['ast'] + team_games['to'])

    team_games['to_ratio'] = 100 * team_games['to'] / (team_games['fga'] + 0.475*team_games['fta'] + \
        team_games['ast'] + team_games['to'])

    team_games['true_shoot'] = 100 * team_games['score'] / (2 * (team_games['fga'] + 0.475*team_games['fta']))
    
    team_games['eff_fg'] = 100 * (team_games['fgm'] + 0.5 * team_games['fgm3']) / team_games['fga']
    
    

    # only include games in which both teams made it to the tourney...
#     team_games = team_games.merge(seeds[['season','teamid']])
#         
#     opp_seeds = seeds.rename(columns={"teamid":"opp_teamid"})
#     team_games = team_games.merge(opp_seeds[['season','opp_teamid']])
    print("get_team_games...done")

    return team_games
    
def get_team_season():
    print("get_team_season...")
    team_games = get_team_games()    
    
    # aggregate by taking the average of stats per season/team
    wl_gb = team_games.groupby(['season','teamid'])
    
    team_season = wl_gb.agg({"win": lambda x:np.mean(x)}).rename(columns={"win":"winpct"})

    def apply_func(gb, col, newcol, fn):
        return gb.agg({col: fn}).rename(columns={col:newcol})


    for col in base_stat_names:
        # add the team's stats
        team_season = team_season.merge(apply_func(wl_gb, col, col, np.mean),
            left_index=True, right_index=True)
        # add the team's opponent's stats
        team_season = team_season.merge(apply_func(wl_gb, "opp_"+col, "opp_"+col, np.mean),
            left_index=True, right_index=True)
    
    for col in ['off_rating','ast_ratio']:
        # add the team's stats
        team_season = team_season.merge(apply_func(wl_gb, col, col, np.mean),
            left_index=True, right_index=True)

    print("get_team_season...done")

    return team_season.reset_index()

def get_rolling_team_games():
    print("get_rolling_team_games...")
    team_games = get_team_games()
    
    # aggregate using a rolling window up-til game day...    
    cols = ['off_rating','def_rating','ast_ratio','to_ratio','true_shoot','eff_fg']
    team_games_shift = team_games.sort_values(['teamid','season','gamedate'])
        
    
    def roll_on_groups(x):
        # shifting takes the current game out of the window
        x[cols] = x[cols].shift(1)
        res = x.rolling(7,on='gamedate',min_periods=5).median()
        return res
        
    
#     print(team_games_shift)
    team_games_shift = team_games_shift.reset_index()
    wl_gb = team_games_shift[['teamid','season','gamedate','win']+cols].groupby(['season','teamid'])
    wl_gb_r = wl_gb[['gamedate']+cols].apply(roll_on_groups)
    wl_gb_r = wl_gb_r.reset_index()
    wl_gb_r = wl_gb_r.drop(['index'],axis=1)
    wl_gb_r = wl_gb_r.dropna()
    team_games_shift = team_games_shift[['teamid','opp_teamid','season','win']].merge(wl_gb_r,left_index=True,right_index=True)
#     wl_gb_r = wl_gb_r.rename(columns={'win':'winpct','opp_win':'opp_winpct'})
    
    print("get_rolling_team_games...done")

    return team_games_shift

def get_rolling_team_season():
    print("get_rolling_team_season...")

    team_games = get_rolling_team_games()
    
    
    wl_gb_last_game = team_games.groupby(['season','teamid']).last().reset_index()
    wl_gb_last_game.dropna(inplace=True);
    
    print("get_rolling_team_season...done")

    return wl_gb_last_game

if __name__ == '__main__':
    get_rolling_team_games()