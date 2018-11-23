import pandas as pd
import numpy as np
import dbConnector

con = dbConnector.getConnection()

team_data = pd.read_sql('''select team_name,team_id,group_concat(match_id) as matchIds,avg(team_rating) as avg_rating from teamDetails

group by team_name''', con)

print(team_data.columns.values)

X = [];

for j in team_data.iterrows():

    for k in j[1]["matchIds"].split(","):

        temp = []

        all_players = pd.read_sql('''select group_concat(player_name) as p_names

         from (select distinct player_name

         from playerperformance

         where player_position_info not like '%Sub%' and team_id =''' + str(
            j[1]["team_id"]) + ''' and match_id=''' + str(k) + ''') tmp

         group by player_name''', con)

        temp = []

        temp.append(j[1]["avg_rating"])

        playerstats = pd.read_sql('''select Volleys

        Vision,

        Strength,

        `Standing tackle`,

        Stamina,

        `Sprint speed`,

        `Sliding tackle`,

        `Shot power`,

        `Short passing`,

        Reactions,

        Penalties,

        Marking,

        `Long shots`,

        `Long passing`,

        Jumping,

        Interceptions,

        `Heading accuracy`,

        `GK reflexes`,

        `GK positioning`,

        `GK kicking`,

        `GK handling`,

        `GK diving`,

        `Free kick accuracy`,

        Finishing,

        Dribbling,

        Curve,

        Crossing,

        Composure,

        `Ball control`,

        Balance,

        Agility,

        Aggression,

        Acceleration,

        consistency,overall

        from playermatchstatstable

        where player_name in (''' + ",".join('"{0}"'.format(w) for w in all_players["p_names"]) + ''')''', con)
        '''
        print('type(playerstats): ', type(playerstats))
        print('playerstats.values.__len__(): ',playerstats.values.__len__())
        print('playerstats.columns.values: ',playerstats.columns.values)
        print('playerstats.values: ', playerstats.values)
        print(str(j[1]["team_id"]), '-', str(k))
        '''
        #print('playerstats.values: ', playerstats.values.tolist())
        #print('playerstats.valuesT.tolist(): ', playerstats.values.T.tolist())
        #for p in playerstats.values.T.tolist():
        #print('playerstats.values.tolist().__len__(): ',playerstats.values.T.tolist().__len__() * 11 + 1)

        for p in playerstats.values.T.tolist():
            for x in p:
                temp.append(x)


        if temp.__len__() < 375:
            while temp.__len__() != 375:
                temp.append(0)

        #print('temp.__len__(): ',temp.__len__())

        X.append(temp)


y = []
x = np.zeros((X.__len__(), X[0].__len__() - 1))
i = 0
for l in X:
    y.append(l[0])
    x[i,:] = l[1:]
    i += 1

print('type(y): ', type(y))
print('type(x): ', type(x))
print('np.shape(y): ', np.shape(y))
print('np.shape(x): ', np.shape(x))

U = np.array(X);
print(type(U))
print('type(U[0])', type(U[0]))
print('np.shape(U): ', np.shape(U))
print('U[0] len', U[0].__len__())
print('U[1] len', U[1].__len__())
print('U[2] len', U[2].__len__())
print('type(np.mat(X)): ',type(np.mat(X)))
print('np.shape(np.mat(X)) ',np.shape(np.mat(X)))

yTraining = U[:,0]
xTraining = U[:,1:]

print('np.shape(xTraining): ', np.shape(xTraining))
print('np.shape(yTraining): ', np.shape(yTraining))

#print('-- U columns:\n', U.columns.values)