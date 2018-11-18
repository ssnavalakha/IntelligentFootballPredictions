import pandas as pd
import dbConnector
import numpy as np
import linearRegression
from sklearn.linear_model import LinearRegression

con = dbConnector.getConnection()

def getTrainingSet(playerData):

    rowsToIgnore = ['id', 'player_fifa_api_id', 'player_api_id', 'date', 'overall_rating','preferred_foot', 'attacking_work_rate', 'defensive_work_rate']

    reducedPlayerData = playerData.drop(rowsToIgnore, axis=1)

    print(reducedPlayerData.columns.values)

    return np.mat(reducedPlayerData)

def getTestSet(playerData):

    testColumnIndex = playerData.columns.get_loc('overall_rating')

    playerTestData = playerData[playerData.columns[testColumnIndex:testColumnIndex+1]]

    return np.mat(playerTestData)

player_data = pd.read_sql("SELECT * FROM player_attributes limit 200;", con)
print(player_data.columns.values)

rows = ['id','player_fifa_api_id','player_api_id','date']

player_data.dropna(subset = rows, inplace = True)

xTraining = getTrainingSet(player_data.head(150))
yTraining = getTestSet(player_data.head(150))

xTest = getTrainingSet(player_data.tail(50))
yTest = getTestSet(player_data.tail(50))

linearRegression.linearReg(xTraining, yTraining, xTest, yTest)

reg = LinearRegression().fit(xTraining, yTraining)
reg.score(xTraining, yTraining)
print('reg.predict(xTest)\n',reg.predict(xTest))
print(yTest)

def calculateError(estimation, Y):

    error = []
    for i in range(0, Y.__len__()):
        er = estimation[i] - Y[i]
        error.append(er * er)

    return sum(error) / Y.__len__()

print('sklearn training: ', calculateError(reg.predict(xTraining), yTraining))
print('sklearn testing: ', calculateError(reg.predict(xTest), yTest))



lambdaList = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1]
kList = [2, 10, 20]

print('\n')
linearRegression.ridgeRegression(xTraining, yTraining, xTest, yTest, lambdaList, kList)
