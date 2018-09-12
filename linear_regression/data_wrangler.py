#!/usr/bin/python
"""
    All data importing/cleaning code goes here.
"""

import numpy as np

def loadTrainingData():
    features = np.genfromtxt("data/hw2.train", delimiter=",")
    labels = features[:,-1]
    features = features[:,:-1]
    return features, labels

def loadTestingData():
    features = np.genfromtxt("data/hw2.test", delimiter=",")
    labels = features[:,-1]
    features = features[:,:-1]
    return features, labels

def loadHousingData():
    pass

def shuffleData(ordered_features, ordered_labels):
    """
        Expects two numpy arrays, features and labels, such that 
        ordered_features[0] has a label in ordered_labels[0].

        this function returns (features,labels) with all of the rows shuffled.
    """
    data = np.column_stack((ordered_features, ordered_labels))
    np.random.shuffle(data)
    labels = data[:, -1]
    features = data[:,:-1]
    return features, labels

def splitDataKFolds(unsplitFeatures, unsplitLabels, k):
    """
        Splits data into k folds of data, returning a list of feature and label
        sets.
        So, if there are 100 rows of features, and 100 labels, and this function
        is called with k = 5, it will return the lists features and labels, 
        which each contain 5 arrays of features and labels respectively. 

        It will be expected that features[0][2] will correspond with labels[0][2]
    """
    if k <= 1:
        return unsplitFeatures, unsplitLabels
    features, labels = [], []
    split = int(len(unsplitFeatures) / k)
    for i in range(0,k):
        features.append(unsplitFeatures[(split*i):(split*(i+1))])
        labels.append(unsplitLabels[(split*i):(split*(i+1))])
    return features, labels

def mergeDatasets(featureList, labelList):
    """
        Assumes that featureList and labelList contain a list of datasets, 
        such that featureList[i] corresponds to labelList[i].
        Should "flatten" the given lists, and return a single list of features
        and a single list of labels.
    """
    features = [y for x in featureList for y in x]
    labels = [y for x in labelList for y in x]
    return features, labels
