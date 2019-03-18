'''
Created on Feb 17, 2019

@author: jim

Reformat the coaches file to have a line for each day. To facilitate joining in pandas.
'''

from csv import DictReader
from csv import DictWriter

if __name__ == '__main__':
    columns = ['season','teamid','day','coach']
    with open("./data/TeamCoachesPerDay.csv", mode='w') as o:
        writer = DictWriter(o, columns)
        writer.writeheader()
        with open("./data/TeamCoaches.csv") as f:
            rdr = DictReader(f)
            for c in rdr:
                for i in range(int(c['firstdaynum']),int(c['lastdaynum'])+1):
                    writer.writerow({
                        'season': c['season'],
                        'teamid': c['teamid'],
                        'day': i,
                        'coach': c['coach']
                        })
    pass