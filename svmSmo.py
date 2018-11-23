import numpy as np
import scipy.io as sio
from sklearn import svm


# Ref - http://cs229.stanford.edu/materials/smo.pdf


def checkAccuracy(prediction, yTest):
    'Get percentage of correct predictions'
    correctCtr = 0
    for i in range(0, yTest.__len__()):
        if prediction[i] == yTest[i]:
            correctCtr += 1

    correctCtr = correctCtr / yTest.__len__()

    return correctCtr * 100


def getJRandom(i, m):
    'Get j not equal to i'
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def getLandH(Y, alphas, i, j, C):
    yi = Y[0, i]
    yj = Y[0, j]
    ai = alphas[i, 0]
    aj = alphas[j, 0]

    L = 0
    H = 0

    if yi != yj:
        L = max(0, aj - ai)
        H = min(C, C + aj - ai)
    else:
        L = max(0, aj + ai - C)
        H = min(C, ai + aj)

    return L, H


def getPrediction(X_tst, X, Y, alphas, b):
    predList = []
    m = np.shape(X_tst)[0]
    for i in range(0, m):

        pred = np.dot(np.multiply(alphas, Y.transpose()).transpose(), np.dot(X, X_tst[i, :].transpose()))  # fx
        pred += b

        if pred < 0:
            predList.append(0)  # -1
        else:
            predList.append(1)

    return predList


def getTargetVariables(oldY):
    'Convert Y from 0,1 to -1,1'
    Y = []
    for i in range(0, oldY.__len__()):
        if oldY[i] == 0:
            Y.append(-1)
        else:
            Y.append(1)
    return Y


def smo(X_trn, Y_trn, C, maxPasses, tolerance):
    # Init
    X = np.mat(X_trn)
    Y = np.mat(Y_trn)
    m, n = np.shape(X)
    alphas = np.mat(np.zeros((m, 1)))
    b = 0
    passes = 0

    while passes < maxPasses:

        numChangedAlphas = 0

        for i in range(0, m):

            fx = np.dot(np.multiply(alphas, Y.transpose()).transpose(), np.dot(X, X[i, :].transpose())) + b

            ei = fx - Y[0, i]
            # print('Error: ', ei)

            yiEi = Y[0, i] * ei

            if ((yiEi < -tolerance) and (alphas[i, 0] < C)) or ((yiEi > tolerance) and (alphas[i, 0] > C)):

                j = getJRandom(i, m)

                fxj = np.dot(np.multiply(alphas, Y.transpose()).transpose(), np.dot(X, X[j, :].transpose())) + b
                ej = fxj - Y[0, j]

                # save old alphas
                alpha_i_old = alphas[i, 0]
                alpha_j_old = alphas[j, 0]

                # get L and H
                L, H = getLandH(Y, alphas, i, j, C)

                if L == H:
                    # print('--- L=H')
                    continue

                n = 2.0 * np.dot(X[i, :], X[j, :].transpose()) - np.dot(X[i, :], X[i, :].transpose()) - np.dot(X[j, :],
                                                                                                               X[j,
                                                                                                               :].transpose())

                if n >= 0:
                    # print('--- n >= 0')
                    continue

                alphas[j, 0] = alphas[j, 0] - ((Y[0, j] * (ei - ej)) / n)

                # clip aj
                if alphas[j, 0] > H:
                    alphas[j, 0] = H
                elif alphas[j, 0] < L:
                    alphas[j, 0] = L

                if abs(alphas[j, 0] - alpha_j_old) < 0.00001:
                    # print('--- < 0.00001')
                    continue

                # set ai from aj
                alphas[i, 0] = alphas[i, 0] + Y[0, i] * Y[0, j] * (alpha_j_old - alphas[j, 0])

                # offset
                b1 = b - ei - (Y[0, i] * (alphas[i, 0] - alpha_i_old) * np.dot(X[i, :], X[i, :].transpose())) - (
                            Y[0, j] * (alphas[j, 0] - alpha_j_old) * np.dot(X[i, :], X[j, :].transpose()))
                b2 = b - ej - (Y[0, i] * (alphas[i, 0] - alpha_i_old) * np.dot(X[i, :], X[j, :].transpose())) - (
                            Y[0, j] * (alphas[j, 0] - alpha_j_old) * np.dot(X[j, :], X[j, :].transpose()))

                if ((0 < alphas[i, 0]) and (alphas[i, 0] < C)):
                    b = b1
                elif ((0 < alphas[j, 0]) and (alphas[j, 0] < C)):
                    b = b2
                else:
                    b = (b1 + b2) / 2

                numChangedAlphas += 1

        if numChangedAlphas == 0:
            passes += 1
        else:
            passes = 0

    return alphas, b


def mySvm(X, Y, cList, X_tst, Y_tst):
    maxPasses = 1000
    tolerance = 0.01

    # print C - Acccuracy for each

    results = []

    for c in cList:
        alphas, b = smo(X, Y, c, maxPasses, tolerance)

        pred = getPrediction(np.mat(X_tst), np.mat(X), np.mat(Y), alphas, b)  # 0,1

        accuracy = checkAccuracy(pred, Y_tst)

        t = (c, accuracy)

        results.append(t)

    for t in results:
        print('C = ', t[0], ' - Accuracy: ', t[1], '%')
    # print('\n')

    return


def skLearnSvm(X, Y, X_tst, Y_tst):
    clf = svm.SVC(gamma='scale')
    clf.fit(X, Y)

    Y_test = getTargetVariables(Y_tst)
    print('sklearn accuracy: ', checkAccuracy(clf.predict(X_tst), Y_test), '\n')

    return

def getTargetVariablesTest(oldY):
    'Convert Y from -1,0,1 to -1,1'
    Y = []
    for i in range(0, oldY.__len__()):
        if (oldY[i] == 0) or oldY[i] == -1:
            Y.append(-1)
        else:
            Y.append(1)
    return Y


def customSvm(X_trn, Y_trn, X_tst, Y_tst):

    #cList = [0.1, 0.2, 0.3, 1, 3, 5, 10]
    cList = [0.1]

    X = X_trn
    Y = getTargetVariablesTest(Y_trn)

    print('\nSVM for given data:\n')

    skLearnSvm(X, Y, X_tst, Y_tst)

    mySvm(X, Y, cList, X_tst, Y_tst)

    skLearnSvm(X, Y, X_tst, Y_tst)


    return


def main():
    # Reg param
    cList = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 5, 10]

    dataSet = sio.loadmat('./dataset1.mat', squeeze_me=True)
    # Y_tst, X_trn Y_trn  X_tst

    X_trn = dataSet['X_trn']
    Y_trn = dataSet['Y_trn']
    X_tst = dataSet['X_tst']
    Y_tst = dataSet['Y_tst']

    X = X_trn
    Y = getTargetVariables(Y_trn)

    print('SVM for dataset1:\n')
    mySvm(X, Y, cList, X_tst, Y_tst)

    skLearnSvm(X, Y, X_tst, Y_tst)

    #############

    dataSet = sio.loadmat('./dataset2.mat', squeeze_me=True)
    # Y_tst, X_trn Y_trn  X_tst

    X_trn = dataSet['X_trn']
    Y_trn = dataSet['Y_trn']
    X_tst = dataSet['X_tst']
    Y_tst = dataSet['Y_tst']

    X = X_trn
    Y = getTargetVariables(Y_trn)

    print('SVM for dataset2:\n')
    mySvm(X, Y, cList, X_tst, Y_tst)

    skLearnSvm(X, Y, X_tst, Y_tst)

    return


#main()


