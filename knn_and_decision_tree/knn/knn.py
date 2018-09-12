#!/usr/bin/python3
from math import *
from operator import itemgetter
from random import choice
from sklearn import preprocessing

def distance(pointA, pointB):
    """
        returns the euclidean distance between two points.
        Assumes that pointA and pointB are test points from the dataset
        with an equal number of features.
    """
    if len(pointA) != len(pointB):
        return -1
    else:
        return sqrt(sum([(pointA[i] - pointB[i]) ** 2 for i in range(len(pointA))]))

def getLabel(labels):
    """
        returns the most common label for a set of labels. If there's a tie in the top
        two labels, it picks one at random.
    """
    elems = {}
    for l in labels:
        if l not in elems.keys():
            elems[l] = 1
        else:
            elems[l] += 1
    counts = sorted(elems.values(), reverse=True)
    if len(counts) > 1 and counts[0] == counts[1]:
        return choice(list(elems.keys()))
    return sorted(elems, key=elems.get, reverse=True)[0]

def kNN(testPoints, k, trainingPoints, trainingLabels):
    """
        Accepts:
            testPoint:
                Point or multiple points that need to be checked against the 
                the trainingPoints.
            k:
                The number of nearest points to compare the testPoint to.
            trainingPoints:
                The training dataset to check against.
            trainingLabels:
                The set of labels that correspond to the trainingPoints.
        Returns:
            predictedLabel(s):
                A set of predicted labels for each testPoint provided. The
                predicted label will be determined by the most represented
                label in the k-nearest points. 
    """

    # check if testPoint is a single point, or a set of points
    # not sure how to do this. Maybe check if the instance is a list, then the first element is a list?
    trainingPoints_s = preprocessing.scale(trainingPoints)
    testPoints_s = preprocessing.scale(testPoints)
    labels_pred = []
    for point_test in testPoints_s:
        point_distances = []
        for ind, point_train in enumerate(trainingPoints_s):
            d = distance(point_test, point_train)
            point_distances.append(tuple([d,ind]))
        point_distances = sorted(point_distances, key=itemgetter(0))
        k_neighbors_indx = [y for x,y in point_distances[0:k]]
        k_labels = [trainingLabels[i] for i in k_neighbors_indx]
        labels_pred.append(getLabel(k_labels))
    return labels_pred
