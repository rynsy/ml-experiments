#!/usr/bin/python3
import unittest
from math import *
from sklearn import neighbors as n
from sklearn import tree
from .context import *

class TestMisclassification(unittest.TestCase):
    def test_misclassification_knn_1(self):
        features_all, labels_all = loadSyntheticData(1)
        features_all, labels_all = shuffleData(features_all, labels_all)
        features_split, labels_split = splitDataKFolds(features_all, labels_all, 5)
        misclassified_sklearn = []
        misclassified_myknn = []
        for k in range(1,11):
            misclassified_sklearn_folds = []
            misclassified_myknn_folds = []
            for i in range(len(features_split)):
                features_test, labels_test= features_split[i], labels_split[i]
                features_train, labels_train = mergeDatasets(features_split[:i] + features_split[i+1:], labels_split[:i] + labels_split[i+1:])               
                neighbs = n.KNeighborsClassifier(n_neighbors=k)
                neighbs.fit(features_train, labels_train)
                labels_pred = neighbs.predict(features_test)
                misclassified_sklearn_folds.append(misclassification(labels_pred, labels_test))
                labels_pred = kNN(features_test, k, features_train, labels_train)
                misclassified_myknn_folds.append(misclassification(labels_pred, labels_test))
            misclassified_sklearn.append(misclassified_sklearn_folds)
            misclassified_myknn.append(misclassified_myknn_folds)
        e_avg = 0.0
        k_sklearn, e_sklearn = 0, 100.0
        for i, m in enumerate(misclassified_sklearn,1):
            e_avg = sum(m) / len(m)
            if e_avg <= e_sklearn:
                e_sklearn = e_avg
                k_sklearn = i
        k_myknn, e_myknn = 0, 100.0
        for i, m in enumerate(misclassified_myknn,1):
            e_avg = sum(m) / len(m)
            if e_avg <= e_myknn:
                e_myknn = e_avg
                k_myknn = i
        self.assertEqual(k_sklearn, k_myknn, msg="Different values of k. k_sklearn: %d k_myknn: %d" % (k_sklearn, k_myknn))
        self.assertAlmostEqual(e_sklearn, e_myknn, msg="Error off. e_sklearn:%f e_myknn:%f" % (e_sklearn, e_myknn))

    def test_misclassification_knn_2(self):
        features_all, labels_all = loadSyntheticData(2)
        features_all, labels_all = shuffleData(features_all, labels_all)
        features_split, labels_split = splitDataKFolds(features_all, labels_all, 5)
        misclassified_sklearn = []
        misclassified_myknn = []
        for k in range(1,11):
            misclassified_sklearn_folds = []
            misclassified_myknn_folds = []
            for i in range(len(features_split)):
                features_test, labels_test= features_split[i], labels_split[i]
                features_train, labels_train = mergeDatasets(features_split[:i] + features_split[i+1:], labels_split[:i] + labels_split[i+1:])               
                neighbs = n.KNeighborsClassifier(n_neighbors=k)
                neighbs.fit(features_train, labels_train)
                labels_pred = neighbs.predict(features_test)
                misclassified_sklearn_folds.append(misclassification(labels_pred, labels_test))
                labels_pred = kNN(features_test, k, features_train, labels_train)
                misclassified_myknn_folds.append(misclassification(labels_pred, labels_test))
            misclassified_sklearn.append(misclassified_sklearn_folds)
            misclassified_myknn.append(misclassified_myknn_folds)
        e_avg = 0.0
        k_sklearn, e_sklearn = 0, 100.0
        for i, m in enumerate(misclassified_sklearn,1):
            e_avg = sum(m) / len(m)
            if e_avg <= e_sklearn:
                e_sklearn = e_avg
                k_sklearn = i
        k_myknn, e_myknn = 0, 100.0
        for i, m in enumerate(misclassified_myknn,1):
            e_avg = sum(m) / len(m)
            if e_avg <= e_myknn:
                e_myknn = e_avg
                k_myknn = i
        self.assertEqual(k_sklearn, k_myknn, msg="Different values of k. k_sklearn: %d k_myknn: %d" % (k_sklearn, k_myknn))
        self.assertAlmostEqual(e_sklearn, e_myknn, msg="Error off. e_sklearn:%f e_myknn:%f" % (e_sklearn, e_myknn))
    
    def test_misclassification_knn_3(self):
        features_all, labels_all = loadSyntheticData(3)
        features_all, labels_all = shuffleData(features_all, labels_all)
        features_split, labels_split = splitDataKFolds(features_all, labels_all, 5)
        misclassified_sklearn = []
        misclassified_myknn = []
        for k in range(1,11):
            misclassified_sklearn_folds = []
            misclassified_myknn_folds = []
            for i in range(len(features_split)):
                features_test, labels_test= features_split[i], labels_split[i]
                features_train, labels_train = mergeDatasets(features_split[:i] + features_split[i+1:], labels_split[:i] + labels_split[i+1:])               
                neighbs = n.KNeighborsClassifier(n_neighbors=k)
                neighbs.fit(features_train, labels_train)
                labels_pred = neighbs.predict(features_test)
                misclassified_sklearn_folds.append(misclassification(labels_pred, labels_test))
                labels_pred = kNN(features_test, k, features_train, labels_train)
                misclassified_myknn_folds.append(misclassification(labels_pred, labels_test))
            misclassified_sklearn.append(misclassified_sklearn_folds)
            misclassified_myknn.append(misclassified_myknn_folds)
        e_avg = 0.0
        k_sklearn, e_sklearn = 0, 100.0
        for i, m in enumerate(misclassified_sklearn,1):
            e_avg = sum(m) / len(m)
            if e_avg <= e_sklearn:
                e_sklearn = e_avg
                k_sklearn = i
        k_myknn, e_myknn = 0, 100.0
        for i, m in enumerate(misclassified_myknn,1):
            e_avg = sum(m) / len(m)
            if e_avg <= e_myknn:
                e_myknn = e_avg
                k_myknn = i
        self.assertEqual(k_sklearn, k_myknn, msg="Different values of k. k_sklearn: %d k_myknn: %d" % (k_sklearn, k_myknn))
        self.assertAlmostEqual(e_sklearn, e_myknn, msg="Error off. e_sklearn:%f e_myknn:%f" % (e_sklearn, e_myknn))
    
    def test_misclassification_knn_4(self):
        features_all, labels_all = loadSyntheticData(4)
        features_all, labels_all = shuffleData(features_all, labels_all)
        features_split, labels_split = splitDataKFolds(features_all, labels_all, 5)
        misclassified_sklearn = []
        misclassified_myknn = []
        for k in range(1,11):
            misclassified_sklearn_folds = []
            misclassified_myknn_folds = []
            for i in range(len(features_split)):
                features_test, labels_test= features_split[i], labels_split[i]
                features_train, labels_train = mergeDatasets(features_split[:i] + features_split[i+1:], labels_split[:i] + labels_split[i+1:])               
                neighbs = n.KNeighborsClassifier(n_neighbors=k)
                neighbs.fit(features_train, labels_train)
                labels_pred = neighbs.predict(features_test)
                misclassified_sklearn_folds.append(misclassification(labels_pred, labels_test))
                labels_pred = kNN(features_test, k, features_train, labels_train)
                misclassified_myknn_folds.append(misclassification(labels_pred, labels_test))
            misclassified_sklearn.append(misclassified_sklearn_folds)
            misclassified_myknn.append(misclassified_myknn_folds)
        e_avg = 0.0
        k_sklearn, e_sklearn = 0, 100.0
        for i, m in enumerate(misclassified_sklearn,1):
            e_avg = sum(m) / len(m)
            if e_avg <= e_sklearn:
                e_sklearn = e_avg
                k_sklearn = i
        k_myknn, e_myknn = 0, 100.0
        for i, m in enumerate(misclassified_myknn,1):
            e_avg = sum(m) / len(m)
            if e_avg <= e_myknn:
                e_myknn = e_avg
                k_myknn = i
        self.assertEqual(k_sklearn, k_myknn, msg="Different values of k. k_sklearn: %d k_myknn: %d" % (k_sklearn, k_myknn))
        self.assertAlmostEqual(e_sklearn, e_myknn, msg="Error off. e_sklearn:%f e_myknn:%f" % (e_sklearn, e_myknn))
    
    def test_misclassification_dt_1(self):
        features_all, labels_all = loadSyntheticData(1)
        features_all, labels_all = shuffleData(features_all, labels_all)
        data = np.column_stack((features_all, labels_all))
        clf = tree.DecisionTreeClassifier(max_depth=3)
        clf = clf.fit(features_all, labels_all)
        myTree = constructTree(data)
        myResults = predict(data, myTree)
        results = clf.predict(features_all)
        myMissed = misclassification(myResults, labels_all)
        missed = misclassification(results, labels_all)
        self.assertAlmostEqual(missed, myMissed, msg="Error off")
    
    def test_misclassification_dt_2(self):
        features_all, labels_all = loadSyntheticData(2)
        features_all, labels_all = shuffleData(features_all, labels_all)
        data = np.column_stack((features_all, labels_all))
        clf = tree.DecisionTreeClassifier(max_depth=3)
        clf = clf.fit(features_all, labels_all)
        myTree = constructTree(data)
        myResults = predict(data, myTree)
        results = clf.predict(features_all)
        myMissed = misclassification(myResults, labels_all)
        missed = misclassification(results, labels_all)
        self.assertAlmostEqual(missed, myMissed, msg="Error off")
    
    def test_misclassification_dt_3(self):
        features_all, labels_all = loadSyntheticData(3)
        features_all, labels_all = shuffleData(features_all, labels_all)
        data = np.column_stack((features_all, labels_all))
        clf = tree.DecisionTreeClassifier(max_depth=3)
        clf = clf.fit(features_all, labels_all)
        myTree = constructTree(data)
        myResults = predict(data, myTree)
        results = clf.predict(features_all)
        myMissed = misclassification(myResults, labels_all)
        missed = misclassification(results, labels_all)
        self.assertAlmostEqual(missed, myMissed, msg="Error off")
    
    def test_misclassification_dt_4(self):
        features_all, labels_all = loadSyntheticData(4)
        features_all, labels_all = shuffleData(features_all, labels_all)
        data = np.column_stack((features_all, labels_all))
        clf = tree.DecisionTreeClassifier(max_depth=3)
        clf = clf.fit(features_all, labels_all)
        myTree = constructTree(data)
        myResults = predict(data, myTree)
        results = clf.predict(features_all)
        myMissed = misclassification(myResults, labels_all)
        missed = misclassification(results, labels_all)
        self.assertAlmostEqual(missed, myMissed, msg="Error off")
