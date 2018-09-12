#!/usr/bin/python3
"""
    This file will contain a valid solution to this assignment using the
    sklearn package. This is mostly for testing and checking my implementation
    of these algorithms. 
"""
from data_wrangler import *
from cross_validator import *
from decision_tree import *
from sketch_artist import *
from knn import *
from sklearn import neighbors as n
from time import sleep
import numpy as np

CV_FOLDS = 5

def calculate_k():
    global CV_FOLDS
    optimal_values = []
    print("Running kNN with %d-fold cross validation" % (CV_FOLDS))
    for indx in range(1,5):
        print("Processing: synthetic-%d" % (indx))
        features_all, labels_all = loadSyntheticData(indx)
        features_all, labels_all = shuffleData(features_all, labels_all)
        features_split, labels_split = splitDataKFolds(features_all, labels_all, CV_FOLDS)
        misclassified_myknn = []
        for k in range(1,11):
            misclassified_myknn_folds = []
            for i in range(len(features_split)):
                features_test, labels_test= features_split[i], labels_split[i]
                features_train, labels_train = mergeDatasets(features_split[:i] + features_split[i+1:], labels_split[:i] + labels_split[i+1:])               
                labels_pred = kNN(features_test, k, features_train, labels_train)
                misclassified_myknn_folds.append(misclassification(labels_pred, labels_test))
            misclassified_myknn.append(misclassified_myknn_folds)
        e_avg = 0.0
        optimal_k, e_min = 0, 100.0
        for i, m in enumerate(misclassified_myknn,1):
            e_avg = sum(m) / len(m)
            if e_avg <= e_min:
                e_min = e_avg
                optimal_k = i
            print("K: %d \t Averaged Error (all folds): %f \t Error per fold: %s" % (i,e_avg,m))
        print("\nOptimal K for sythetic-%d is: %d, its average error is:%f\n" % (indx, optimal_k, e_min))
        optimal_values.append(optimal_k)
    return optimal_values

k_values = calculate_k()
plotKNN(k_values)
calculateAndPlotDT()
