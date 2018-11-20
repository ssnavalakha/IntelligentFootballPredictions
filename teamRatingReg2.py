import pandas as pd
import numpy as np
import dbConnector
import linearRegression
from sklearn.linear_model import LinearRegression

con = dbConnector.getConnection()

matchQuery = 'SELECT match_id FROM playerperformance p group by match_id limit 110;'
match_ids = pd.read_sql(matchQuery ,con)

# list of match ids
matchList = list(match_ids.values.flatten())
print(matchList)

# test for 1 match
#matchList = [829514]

playerTeamRatingsQuery = '''select q.overall as 'player_rating' , t.team_rating, p.team_id from
    playerperformance p left join playermatchstatstable q on p.player_name = q.player_name
    left join teamdetails t on p.match_id = t.match_id and p.team_id = t.team_id
    where p.match_id = {match_identifier} and player_position_info not like '%sub%' order by p.team_id 
    limit 30;'''  # remove limit later

teamRatingMatrix = np.zeros((matchList.__len__() * 2,23))
print('np.shape(teamRatingMatrix): ', np.shape(teamRatingMatrix))

currentRowCounter = 0

for match in matchList:

    #print('Match_id: ', match)

    # get player ratings and team rating
    ratingsData = pd.read_sql(playerTeamRatingsQuery.format(match_identifier=match), con)
    #print('ratingsData:\n',ratingsData)

    # fill all the null ratings (due to data inconsistencies) with avg ratings
    ratingsData['player_rating'].fillna(ratingsData['player_rating'].mean(), inplace=True)

    # get xTraining : player ratings vector and yTraining: team rating

    # drop extra players
    ind = 0
    for i in range(1, ratingsData.shape[0] - 1):
        if ratingsData['team_id'][i] != ratingsData['team_id'][i-1]:
            ind = i
            break
    #print('ind: ', ind)

    team1Ratings = ratingsData.loc[:10, :]
    team2Ratings = ratingsData.loc[ind:(ind + 10), :]

    #print('team1Ratings: \n', team1Ratings)
    #print('team2Ratings: \n', team2Ratings)

    team1PlyrRatingList = list(team1Ratings['player_rating'].values.flatten())
    team2PlyrRatingList = list(team2Ratings['player_rating'].values.flatten())

    trainingRow1 = team1PlyrRatingList.copy()
    trainingRow2 = team2PlyrRatingList.copy()

    trainingRow1.extend(team2PlyrRatingList)
    trainingRow2.extend(team1PlyrRatingList)

    # append response to rows
    trainingRow1.append(team1Ratings['team_rating'][0])
    trainingRow2.append(team2Ratings['team_rating'][ind])

    teamRatingMatrix[currentRowCounter, :] = trainingRow1
    currentRowCounter += 1
    teamRatingMatrix[currentRowCounter, :] = trainingRow2
    currentRowCounter += 1


#print(teamRatingMatrix)
teamRatingMatrix = np.mat(teamRatingMatrix)

training = teamRatingMatrix[:100,:]
testing = teamRatingMatrix[101:,:]
rows, cols = np.shape(teamRatingMatrix)
xTraining = training[:,0:cols-1]
yTraining = training[:,cols-1]
xTest = testing[:,:cols-1]
yTest = testing[:,cols-1]

linearRegression.linearReg(xTraining, yTraining, xTest, yTest)


print('\n')

reg = LinearRegression().fit(xTraining, yTraining)
reg.score(xTraining, yTraining)
print('reg.predict(xTest)\n',reg.predict(xTest))

def calculateError(estimation, Y):

    error = []
    for i in range(0, Y.__len__()):
        er = estimation[i] - Y[i]
        error.append(er * er)

    return sum(error) / Y.__len__()

print('sklearn training: ', calculateError(reg.predict(xTraining), yTraining))
print('sklearn testing: ', calculateError(reg.predict(xTest), yTest))


l = reg.predict(xTest)
flat_list = [item for sublist in l for item in sublist]

print('\nActual vs pred:')
for x in range(0, flat_list.__len__()):
    print(yTest[x], ' - ', flat_list[x])

