#!/usr/bin/python3
import unittest
from math import *
from .context import *

class TestCrossValidator(unittest.TestCase):
    def test_accuracy(self):
        r = accuracy([0,0,1], [0,0,0])
        self.assertEqual(round(r,3) , 0.667)
        r = accuracy([0,0,0,1], [0,0,0,0])
        self.assertEqual(round(r,2) , 0.75)
        r = accuracy([0,0,0,0], [0,0,0,0])
        self.assertEqual(round(r,0) , 1.0)
        r = accuracy([1,1,1,1], [0,0,0,0])
        self.assertEqual(round(r,0) , 0.0)
