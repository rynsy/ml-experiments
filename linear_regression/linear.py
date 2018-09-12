#!/usr/bin/python
from __future__ import division
from sklearn import preprocessing
import numpy as np

alpha = 0.0005 # learning rate

def fit_lr_normal(data, labels):
    """
        fit a linear regression model using normal equations
        Does not work for singular matricies.
    """
    scaled_data = preprocessing.scale(np.column_stack((data,labels)))
    x = scaled_data[:,:-1]
    y = scaled_data[:,-1]
    xT = x.transpose()
    try:
        t = np.dot(np.dot(np.linalg.inv(np.dot(xT, x)),xT),y)
    except np.linalg.linalg.LinAlgError:
        t = np.ones(np.shape(xT)[0])
    return t

def fit_lr_gd(data, labels):
    """
        fit a linear regression model using gradient descent (regular or stochastic)
        TODO: rewrite to work with multiple columns, without adding bias, etc. 
        Add bias terms to data before calling this function
    """
    global alpha
    scaled_data = preprocessing.scale(np.column_stack((data,labels)))
    x = scaled_data[:,:-1] #select all columns but last
    y = scaled_data[:,-1]
    xT = x.transpose()
    t = np.ones(np.shape(xT)[0])
    i = 0
    cost, prevCost = 1000.0, 0.0
    #while cost - prevCost > 0.0000001:
    while i < 100000:
        h = np.dot(x,t)
        loss = h - y
        prevCost = cost
        cost = np.sum(loss ** 2) / (2 * len(y))
        #print "Iteration %d | Cost: %f" % (i, cost)
        grad = np.dot(xT, loss) / len(y)
        t = t - alpha * grad
        i += 1
    return t

def predict_lr(model, data):
    """
        Using a linear regression model, predict labels for given data
        TODO: Rewrite in terms of multiple theta
    """
    return np.dot(data,model)

def compute_mse(labels_ground_truth, labels_estimated):
    """
        Compute the mean squared error

        TODO: Write this and test cases first.
    """
    return (1 / len(labels_estimated)) * np.sum((labels_estimated - labels_ground_truth) ** 2)
