#!/usr/bin/python3
import unittest
from .context import *

class TestDataWrangler(unittest.TestCase):
    def test_synthetic_loader(self):
        f, l = loadSyntheticData(1)
        self.assertEqual(len(f), 200)
        self.assertEqual(len(l), 200)
        self.assertEqual(len(f[0]), 2)

    def test_data_split_1(self):
        f0, l0 = loadSyntheticData(1)
        f1, l1 = splitDataKFolds(f0, l0, 1)
        self.assertEqual(len(f0), len(f1))
        self.assertEqual(len(l0), len(l1))
        self.assertEqual(len(f0[0]), len(f1[0]))

    def test_data_split_2(self):
        f0, l0 = loadSyntheticData(1)
        f2, l2 = splitDataKFolds(f0, l0, 2)
        self.assertEqual(len(f2), 2)
        self.assertEqual(len(l2), 2)
        self.assertEqual(len(f2[0]) + len(f2[1]), len(f0))
        self.assertEqual(len(l2[0]) + len(l2[1]), len(l0))
        self.assertEqual(f2[0][0][0], f0[0][0])
        self.assertEqual(f2[0][0][1], f0[0][1])
        self.assertEqual(l2[0][0], l0[0])
        self.assertEqual(l2[0][1], l0[1])

    def test_data_split_10(self):
        f0, l0 = loadSyntheticData(1)
        f10, l10 = splitDataKFolds(f0, l0, 10)
        self.assertEqual(len(f10), 10)
        self.assertEqual(len(l10), 10)
        self.assertEqual(sum([len(x) for x in f10]), len(f0))
        self.assertEqual(sum([len(x) for x in l10]), len(l0))
