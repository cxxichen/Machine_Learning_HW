import numpy as np
from utils import *
from run_knn import run_knn
import matplotlib.pyplot as plt

def calClassificationRate(k, train_data, train_target, input_data, intput_target):
    outputTarget = run_knn(k, train_data, train_target, input_data)
    correctTarget = 0
    for i in range(len(intput_target)):
        if (intput_target[i] == outputTarget[i]):
            correctTarget += 1
    return correctTarget / float(len(intput_target))

def plotResult(k, valid, test):
    plt.plot(k, valid, 'g-', label='Validation')
    plt.plot(k, test, 'r--', label='Test')
    plt.xlabel('K')
    plt.ylabel('Classification Rate')
    plt.legend(loc='lower right', numpoints = 1)
    plt.ylim([0.8, 1])
    plt.title('Classfication Rate Vs different value of K')
    plt.show()

if __name__ == '__main__':
    trainData, trainTarget = load_train()
    validData, validTarget = load_valid()
    testData, testTarget = load_test()

    k = [1, 3, 5, 7, 9]
    classificationRate_valid = []
    classificationRate_test = []

    print 'Validation'
    for i in k:
        res = calClassificationRate(i, trainData, trainTarget, validData, validTarget)
        classificationRate_valid.append(res)
        print 'K = ' + repr(i) + ': ' + repr(res)

    print '\nTest'
    for i in k:
        res = calClassificationRate(i, trainData, trainTarget, testData, testTarget)
        classificationRate_test.append(res)
        print 'K = ' + repr(i) + ': ' + repr(res)

    plotResult(k, classificationRate_valid, classificationRate_test)