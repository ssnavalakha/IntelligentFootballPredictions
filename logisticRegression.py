import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Call logisticRef method

def logit(z):
    sigma = 1 / (1 + np.exp(-z))
    return sigma


def likelihood(X, theta, Y):
    'loss - Minimize using gd'
    yRows = np.shape(Y)[0]
    z = X.dot(theta)
    l = logit(z)
    loss = ((-Y * np.log(l) - (1 - Y) * np.log(1 - l))) / yRows
    return loss


def getGradient(X, Y, theta):
    z = X.dot(theta)
    l = logit(z)
    grad = X.transpose().dot(l - Y) / np.shape(Y)[0]
    return grad


def getPrediction(X, theta):
    z = X.dot(theta)
    pred = logit(z)
    roundedPred = []
    for p in pred:
        if p <= 0.5:
            roundedPred.append(0)
        else:
            roundedPred.append(1)
    return roundedPred


def checkAccuracy(xTest, yTest, theta):
    correctCtr = 0  # number of correct predictions
    prediction = getPrediction(xTest, theta)

    for i in range(0, xTest.__len__()):
        if prediction[i] == yTest[i]:
            correctCtr += 1

    correctCtr = correctCtr / prediction.__len__()

    return correctCtr

'''
def checkAccuracySkLearn(xTest, yTest):
    correctCtr = 0  # number of correct predictions
    prediction = getPrediction(xTest, theta)

    for i in range(0, xTest.__len__()):
        if prediction[i] == yTest[i]:
            correctCtr += 1

    correctCtr = correctCtr / prediction.__len__()

    return correctCtr
'''

def calculateThetaGradDesc(X, Y):
    learningRate = 0.1
    theta = np.zeros(np.shape(X)[1])  # d*1

    for i in range(0, 100):
        gradient = getGradient(X, Y, theta)
        theta = theta - (learningRate * gradient)
        loss = likelihood(X, theta, Y)

        l1Error = sum(pow(loss, 2)) / loss.__len__()

        # print('Error: ', l1Error, ' Theta: ', theta)

        # if l1Error < 0.001:
        #    break

    return theta


def logisticRef(xTraining,yTraining,xTest, yTest):
    theta = calculateThetaGradDesc(xTraining, yTraining)
    print('Theta: ', theta)
    print('Training accuracy: ', checkAccuracy(xTraining, yTraining, theta))
    print('Test accuracy: ', checkAccuracy(xTest, yTest, theta))

    # sklearn
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class = 'multinomial').fit(xTraining, yTraining)
    print('clf.predict(xTest): \n', clf.predict(xTest))
    print('clf.predict_proba(X):\n', clf.predict_proba(xTest))
    print(clf.score(xTest, yTest))

    return

def main():
    dataSet = sio.loadmat('.\\dataset3.mat', squeeze_me=True)
    xTraining = dataSet['X_trn']
    yTraining = dataSet['Y_trn']
    xTest = dataSet['X_tst']
    yTest = dataSet['Y_tst']

    print('Logistic regression: \n')

    print('For dataset3.mat')
    theta = calculateThetaGradDesc(xTraining, yTraining)
    print('Theta: ', theta)

    print('Training accuracy: ', checkAccuracy(xTraining, yTraining, theta))
    print('Test accuracy: ', checkAccuracy(xTest, yTest, theta))

    print('\nFor dataset4.mat')

    dataSet = sio.loadmat('.\\dataset4.mat', squeeze_me=True)
    xTraining = dataSet['X_trn']
    yTraining = dataSet['Y_trn']
    xTest = dataSet['X_tst']
    yTest = dataSet['Y_tst']

    theta = calculateThetaGradDesc(xTraining, yTraining)
    print('Theta: ', theta)

    print('Training accuracy: ', checkAccuracy(xTraining, yTraining, theta))
    print('Test accuracy: ', checkAccuracy(xTest, yTest, theta))

    print('\nNote: Accuracy is calculated as (correct predictions / total predictions)')
    # unable to plot hyperplane

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(xTraining, yTraining)
    print('clf.predict(xTest): \n', clf.predict(xTest))
    print('clf.predict_proba(X):\n', clf.predict_proba(xTest))
    print(clf.score(xTest, yTest))

    return


#main()

