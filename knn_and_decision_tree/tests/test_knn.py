#!/usr/bin/python3
import unittest
from math import *
from sklearn import neighbors as n
from .context import *

class TestKnn(unittest.TestCase):
    def test_euclid_small_lists(self):
        p1 = [0]
        p2 = [1]
        d = distance(p1,p2)
        self.assertEqual(d, 1)
        p1 = [0,0]
        p2 = [0,1]
        d = distance(p1,p2)
        self.assertEqual(d, 1)
        p1 = [1,0]
        d = distance(p1,p2)
        self.assertEqual(round(d,3), round(sqrt(2),3))

    def test_get_label(self):
        data = [0,1,0,1,1,0,1]
        self.assertEqual(getLabel(data), 1)
        data = [0,0,1,0,1]
        self.assertEqual(getLabel(data), 0)
        data = [0,0,1,1,1]
        self.assertEqual(getLabel(data), 1)
        data = [1,1,1,0,0]
        self.assertEqual(getLabel(data), 1)
        data = [0,0,0,1,1]
        self.assertEqual(getLabel(data), 0)
        data = [1,1,0,0,0]
        self.assertEqual(getLabel(data), 0)

    def test_knn_manual(self):
        points_train = [[1,0,0], [0,1,0], [0,0,1]]
        points_labels = [1.0, 0.0, 0.0]
        points_test = [[2,0,1]]
        result = kNN(points_test, 1, points_train, points_labels)
        self.assertEqual(result, [1.0])
        result = kNN(points_test, 3, points_train, points_labels)
        self.assertEqual(result, [0.0])
        points_labels = [0.0, 1.0, 1.0]
        result = kNN(points_test, 1, points_train, points_labels)
        self.assertEqual(result, [0.0])
        result = kNN(points_test, 3, points_train, points_labels)
        self.assertEqual(result, [1.0])
       
    def test_knn_random_manual(self):
        """
            Need to check for randomly decided ties
        """
        points_train = [[1,0,0], [0,1,0], [0,0,1]]
        points_labels = [0.0, 1.0, 1.0]
        points_test = [[2,0,1]]
        always_true = True
        for _ in range(100):
            always_true = always_true and ([0.0] == \
                    kNN(points_test, 2, points_train, points_labels))
        self.assertEqual(always_true, False)
