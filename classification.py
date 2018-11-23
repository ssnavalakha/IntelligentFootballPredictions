import pandas as pd
import numpy as np
import dbConnector
import logisticRegression
from sklearn.linear_model import LogisticRegression
import svmSmo

def getTargetVariables(oldY):
    'Convert Y from 0,1,-1 to -1,1'
    Y = []
    for i in range(0, oldY.__len__()):
        if (oldY[i] == 0) or (oldY[i] == -1):
            Y.append(0)
        else:
            Y.append(1)
    return Y

con = dbConnector.getConnection()

match_features = '''SELECT tr.*, tp.goal_diff_home,tp.home_win_perc,tp.home_lose_perc,tp.home_draw_perc,
tpa.goal_diff_away,tpa.away_win_perc,tpa.away_lose_perc,tpa.away_draw_perc,
s.full_time_score,
(if( SUBSTRING_INDEX(s.full_time_score, ' : ', 1) > SUBSTRING_INDEX(s.full_time_score, ' : ', -1), 1,
 (if( SUBSTRING_INDEX(s.full_time_score, ' : ', 1) = SUBSTRING_INDEX(s.full_time_score, ' : ', -1), 0, -1)))) 
 as 'match_outcome'
FROM teamratings tr
inner join
teamperformance tp on tr.home_team_id = tp.team_id
inner join
teamperformance tpa on tr.away_team_id = tpa.team_id
inner join
season_match_stats s on tr.match_id = s.match_id and tr.home_team_id = s.home_team_id
;'''

match_data = pd.read_sql(match_features,con)

# print('match_data.columns.values: ', match_data.columns.values)
print('np.shape(match_data): ', np.shape(match_data))

rowsToIgnore = ['match_id','home_team_id','away_team_id','full_time_score']
reducedMatchData = match_data.drop(rowsToIgnore, axis=1)

# print('reducedMatchData.columns.values: ', reducedMatchData.columns.values)
print('np.shape(reducedMatchData): ', np.shape(reducedMatchData))

# get training and testing matrix
rows, columns = np.shape(reducedMatchData)
trainingLimit = rows - int(rows * 0.01)  # 10 percent data for testing

# last column - match_outcome is response (1 - win, 0 - draw, -1 - loss)
reducedMatchData = np.mat(reducedMatchData)
xTraining = reducedMatchData[:trainingLimit, 0:columns-1]
yTraining = reducedMatchData[:trainingLimit, columns-1]
xTest = reducedMatchData[trainingLimit:, 0:columns-1]
yTest = reducedMatchData[trainingLimit:, columns-1]

print('trainingLimit: ', trainingLimit)
print('np.shape(xTraining): ', np.shape(xTraining))
print('np.shape(yTraining): ', np.shape(yTraining))
print('np.shape(xTest): ', np.shape(xTest))
print('np.shape(yTest): ', np.shape(yTest))

# convert test from win vs loss (draw is loss)
yTraining = np.mat(getTargetVariables(yTraining))
yTest = np.mat(getTargetVariables(yTest)) # y in 0 and 1s for logistic regression

# test
xTraining = np.squeeze(np.asarray(xTraining))
yTraining = np.squeeze(np.asarray(yTraining))
xTest = np.squeeze(np.asarray(xTest))
yTest = np.squeeze(np.asarray(yTest))

'''
print('np.shape(xTraining): ',np.shape(xTraining))
print('np.shape(yTraining): ', np.shape(yTraining.transpose()))
print(type(xTraining))
print(type(yTraining))
'''

logisticRegression.logisticReg(xTraining,yTraining.transpose(),xTest, yTest.transpose())

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class = 'multinomial').fit(xTraining, yTraining.transpose())
print('clf.predict(xTest): \n', clf.predict(xTest))
print(yTest)
print('clf.predict_proba(X):\n', clf.predict_proba(xTest))
print(clf.score(xTest, yTest.transpose()))


svmSmo.customSvm(xTraining, yTraining, xTest, yTest)