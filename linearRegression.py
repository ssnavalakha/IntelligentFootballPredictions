import scipy.io as sio
from numpy.linalg import inv
import numpy as np


class RidgeRegResults():  # class to store findings for k fold ridge
    def __init__(self):
        self.k = 0
        self.lamb = 0
        self.testIndex = 0  # to id hold out partition
        self.sseTraining = 0
        self.sseTest = 0
        self.theta = []

    def setResults(self, k, lamb, testIndex, sseTraining, sseTest, theta):
        self.k = k
        self.lamb = lamb
        self.testIndex = testIndex
        self.sseTraining = sseTraining
        self.sseTest = sseTest
        self.theta = theta
        return

    # test method
    def printData(self):
        print('\n-----')
        print(' k = ', self.k, ' testIndex = ', self.testIndex, ' lambda = ', self.lamb, ' sseTest: ',
              self.sseTest, ' sseTraining: ', self.sseTraining)
        print('theta: ', self.theta)
        return


def calculateError(X, theta, Y):
    estimation = X.dot(theta)

    error = []
    for i in range(0, Y.__len__()):
        er = estimation[i] - Y[i]
        error.append(er * er)

    return sum(error) / Y.__len__()


def calculateThetaClosedForm(X, Y):
    X_Transpose = X.transpose()

    theta = inv(X_Transpose.dot(X)).dot(X_Transpose).dot(Y)

    return theta


def getGradient(X, theta, Y, learningRate):
    X_transpose = X.transpose()
    estimation = X.dot(theta)
    error = estimation - Y

    gradient = X_transpose.dot(error) / (2 * Y.__len__())
    gradient = learningRate * gradient

    return gradient


def getConvergence(newTheta, oldTheta):
    'L1 error for thetas in gradient descent'

    diff = newTheta - oldTheta

    l1Norm = 0

    for n in diff:
        l1Norm += abs(n)

    return l1Norm


def calculateThetaRidge(X, Y, lamb):
    X_Transpose = X.transpose()

    # make regularization matrix for lambda
    xCol = np.shape(X)[1]
    regMatrix = np.eye(xCol)
    regMatrix = lamb * regMatrix
    regMatrix_Transpose = regMatrix.transpose()

    theta = inv(X_Transpose.dot(X) + regMatrix_Transpose.dot(regMatrix)).dot(X_Transpose).dot(Y)

    return theta


def calculateThetaGradientDescent(X, yTraining, n):
    maxIterations = 10000
    epsilon = 0.0001

    oldTheta = []
    newTheta = np.ones(n + 1)  # starting with 1s as initial theta

    for i in range(0, maxIterations):

        oldTheta = newTheta

        newTheta = newTheta - getGradient(X, newTheta, yTraining, 0.001)

        error = getConvergence(newTheta, oldTheta)

        # print('Iteration: ', i, ' Convergence: ', error)

        if error <= epsilon:
            print('Converged')
            break

    return newTheta


def getPhi(X, n):
    rows = X.__len__()
    cols = n + 1

    phi = np.ones(shape=(rows, cols))  # all 1s
    phi[:, 1] = X  # 2nd row is X

    for ind in range(2, n + 1):
        phi[:, ind] = pow(phi[:, 1], ind)

    return phi


def ridgeRegression(xTraining, yTraining, xTest, yTest, lambdaList, kList, nList = 0):
    print('Ridge Regression\n')

    #for n in nList:

    findings = []

    for k in kList:

        # partition as per K
        partitionXList = np.array_split(xTraining, k)
        partitionYList = np.array_split(yTraining, k)

        for i in range(0, k):  # take partition i as test
            xTrn = []
            yTrn = []
            xTst = partitionXList[i]
            yTst = partitionYList[i]

            #XTest = getPhi(xTst, n)
            XTest = xTst
            YTest = yTst

            for j in range(0, k):  # add remaining partitions to training
                if i != j:
                    xTrn.extend(partitionXList[j])
                    yTrn.extend(partitionYList[j])

            X = xTrn
            Y = yTrn

            # X, Y, XTest, XTest

            for lamb in lambdaList:
                theta = calculateThetaRidge(X, Y, lamb)
                sseTraining = calculateError(X, theta, Y)
                sseTest = calculateError(XTest, theta, YTest)

                # store findings in object
                resObj = RidgeRegResults()
                resObj.setResults(k, lamb, i, sseTraining, sseTest, theta)
                findings.append(resObj)

    '''
    if not not findings:
        for f in findings:
            f.printData()
    '''
    # get optimal lambda, sseTest, sseTraining, theta - sort on sseTest
    findings = sorted(findings, key=lambda linkObj: linkObj.sseTest)

    print('Findings length = ', findings.__len__())
    for i in findings.__len__():
        print('\nOptimal values: ')
        print('Theta: ', findings[i].theta)
        print('Lambda: ', findings[i].lamb)
        print('Training Error: ', findings[i].sseTraining)
        print('Test Error: ', findings[i].sseTest)
        print('K: ', findings[i].k)
        print('\n')

    return


def linearReg(xTraining, yTraining, xTest, yTest, nList = 0):
    print('Linear Regression\n')

    #for n in nList:
    #print('For n = ', n)

    '''
    X = getPhi(xTraining, n)
    phiXTest = getPhi(xTest, n)
    '''

    X = xTraining

    print('With closed form:')

    theta = calculateThetaClosedForm(X, yTraining)
    print('Theta: ', theta)

    sseTraining = calculateError(X, theta, yTraining)

    print('Error for training: ', sseTraining)

    #sseTest = calculateError(phiXTest, theta, yTest)
    sseTest = calculateError(xTest, theta, yTest)

    print('Error for testing: ', sseTest)
    print('\n')
    '''
    print('With gradient descent:')

    theta = calculateThetaGradientDescent(X, yTraining, n)

    print('Theta: ', theta)

    sseTraining = calculateError(X, theta, yTraining)

    print('Error for training: ', sseTraining)

    sseTest = calculateError(phiXTest, theta, yTest)

    print('Error for testing: ', sseTest)
    print('\n')
    '''
    return


def main():
    # Linear regression

    # load data
    dataSet = sio.loadmat('./dataset1.mat', squeeze_me=True)
    xTraining = dataSet['X_trn']
    yTraining = dataSet['Y_trn']
    xTest = dataSet['X_tst']
    yTest = dataSet['Y_tst']
    n = [2, 3]

    linearReg(xTraining, yTraining, xTest, yTest, n)

    # Ridge regression

    dataSet = sio.loadmat('./dataset2.mat', squeeze_me=True)
    xTraining = dataSet['X_trn']
    yTraining = dataSet['Y_trn']
    xTest = dataSet['X_tst']
    yTest = dataSet['Y_tst']

    lambdaList = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1]
    kList = [2, 10, yTraining.__len__()]
    nList = [2, 5]

    ridgeRegression(xTraining, yTraining, xTest, yTest, lambdaList, kList, nList)

    return


#main()