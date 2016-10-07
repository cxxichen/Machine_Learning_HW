""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    N, M = data.shape
    y = np.zeros((N, 1))
    for i in range(N):
        z = np.dot(data[i, :], weights[:M]) + weights[-1]
        y[i] = sigmoid(z)

    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    ce = 0.0
    correct = 0
    for i in range(len(targets)):
        if (y[i][0] >= 0.5 and targets[i][0] == 0):
            correct += 1
        if (y[i][0] < 0.5 and targets[i][0] == 1):
            correct += 1
    frac_correct = float(correct)/float(len(targets))

    ce += (-1.0 * np.dot(targets.T, np.log(1-y)) - 1.0 * np.dot((1 - targets.T), np.log(y)))[0][0]
    
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # TODO: compute f and df without regularization
        N, M = data.shape
        f = 0.0
        df = np.zeros(((M+1), 1))
        z = [0.0] * N
        mod_data = np.ones((N, M+1))
        mod_data[:, :-1] = data
        z = np.dot(mod_data, weights)

        f = float(np.sum(np.log(1 + np.exp(-z))) + np.dot(np.transpose(targets), z))
        df = np.dot(np.transpose(mod_data), targets - (1- sigmoid(z)))

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    # TODO: Finish this function
    # [...]
    f = 0
    df = 0
    return f, df
