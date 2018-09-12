#!/usr/bin/python3
import unittest
from .context import *

class TestEquations(unittest.TestCase):
    def setUp(self):
        self.data = [['google', 10, False], 
                ['google', 20, True], 
                ['yahoo', 10, True], 
                ['yahoo', 20, True],
                ['altavista', 10, False],
                ['altavista', 20, True]]

    def test_divide_1(self):
        set1, set2 = divide(self.data, 0, 'google')
        test_split_1 = [['google', 10, False], ['google', 20, True]]
        test_split_2 =  [['yahoo', 10, True], 
                         ['yahoo', 20, True],
                         ['altavista', 10, False],
                         ['altavista', 20, True]]
        self.assertEqual(set1, test_split_1)
        self.assertEqual(set2, test_split_2)
        
    def test_divide_2(self):
        set1, set2 = divide(self.data, 1, 11)
        test_split_1 =  [['google', 20, True],
                         ['yahoo', 20, True],
                         ['altavista', 20, True]]
        test_split_2 = [['google', 10, False], 
                        ['yahoo', 10, True], 
                        ['altavista', 10, False]]
        self.assertEqual(set1, test_split_1)
        self.assertEqual(set2, test_split_2)
    
    def test_divide_3(self):
        set1, set2 = divide(self.data, 2, False)
        test_split_1 = [['google', 10, False], 
                        ['altavista', 10, False]]
        test_split_2 = [['google', 20, True],
                        ['yahoo', 10, True], 
                        ['yahoo', 20, True],
                        ['altavista', 20, True]]
        self.assertEqual(set1, test_split_1)
        self.assertEqual(set2, test_split_2)

    def test_unique_1(self):
        set1, set2 = divide(self.data, 0, 'google')
        r1 = unique(set1)
        r2 = unique(set2)
        self.assertEqual(r1[True], 1)
        self.assertEqual(r1[False], 1)
        self.assertEqual(r2[True], 3)
        self.assertEqual(r2[False], 1)

    def test_unique_2(self):
        set1, set2 = divide(self.data, 1, 11)
        r1 = unique(set1)
        r2 = unique(set2)
        self.assertEqual(r1[True], 3)
        self.assertTrue(False not in r1.keys())
        self.assertEqual(r2[True], 1)
        self.assertEqual(r2[False], 2)

