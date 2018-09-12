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

def myknn():
    optimal_k = []
    for indx in range(1,5):
        features_all, labels_all = loadSyntheticData(indx)
        features_all, labels_all = shuffleData(features_all, labels_all)
        features_split, labels_split = splitDataKFolds(features_all, labels_all, 5)
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
        k_myknn, e_myknn = 0, 100.0
        for i, m in enumerate(misclassified_myknn,1):
            e_avg = sum(m) / len(m)
            if e_avg <= e_myknn:
                e_myknn = e_avg
                k_myknn = i
        print("Dataset: %s K: %d Error: %f" % ("synthetic-"+str(indx), k_myknn, e_myknn))
        optimal_k.append(k_myknn)
    return optimal_k

def sklearn_knn():
    optimal_k = []
    for indx in range(1,5):
        features_all, labels_all = loadSyntheticData(indx)
        features_all, labels_all = shuffleData(features_all, labels_all)
        features_split, labels_split = splitDataKFolds(features_all, labels_all, 5)
        misclassified_sklearn = []
        for k in range(1,11):
            misclassified_sklearn_folds = []
            for i in range(len(features_split)):
                features_test, labels_test= features_split[i], labels_split[i]
                features_train, labels_train = mergeDatasets(features_split[:i] + features_split[i+1:], labels_split[:i] + labels_split[i+1:])               
                neighbs = n.KNeighborsClassifier(n_neighbors=k, algorithm='brute')
                neighbs.fit(features_train, labels_train)
                labels_pred = neighbs.predict(features_test)
                misclassified_sklearn_folds.append(misclassification(labels_pred, labels_test))
            misclassified_sklearn.append(misclassified_sklearn_folds)
        e_avg = 0.0
        k_sklearn, e_sklearn = 0, 100.0
        for i, m in enumerate(misclassified_sklearn,1):
            e_avg = sum(m) / len(m)
            if e_avg <= e_sklearn:
                e_sklearn = e_avg
                k_sklearn = i
        print("Dataset: %s K: %d Error: %f" % ("synthetic-"+str(indx), k_sklearn, e_sklearn))
        optimal_k.append(k_sklearn)
    return optimal_k

def calculateOptimalDepths():
    global POINT_NUM
    plot_colors = "br"
    misclassified_datasets = []
    optimal_depths = []
    for pindx in range(1,5):
        print("Processing dataset synthetic-%d" % (pindx))
        features, labels = loadSyntheticData(pindx)
        misclassified_depths = []
        for d in range(1,11):
            myTree = constructTree(np.column_stack((features, labels)), max_depth=d)
            labels_pred = predict(np.column_stack((features,labels)), myTree)
            m_err = misclassification(labels_pred, labels)
            misclassified_depths.append(m_err)
        misclassified_datasets.append(misclassified_depths)
        e_avg = 0.0
        depth, error = 0, 100.0
        for i, m in enumerate(misclassified_datasets,1):
            e_avg = sum(m) / len(m)
            if e_avg <= error:
                error = e_avg
                depth = i
            print("Depth: %d \t Averaged Error (all folds): %f \t Error per fold: %s" % (i,e_avg,m))
        print("Dataset: %s Depth: %d Error: %f" % ("synthetic-"+str(pindx), depth, error))
        optimal_depths.append(depth)
    return optimal_depths

calculateOptimalDepths()
