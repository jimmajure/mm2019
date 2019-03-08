### March Madness 2019

## Initial Approach

1. Summarize the `RegularSeasonDetailedResults.csv` for each season/team.
2. Do simple ML/regression to attempt to predict wins.
3. Summarize the data within a window of the game being predicted.
4. Add information about the players.


`data['gamedate'] = pd.to_datetime(data['dayzero'], format='%m/%d/%Y') + pd.to_timedelta(data['day'], unit='d')`

## Generated data files...