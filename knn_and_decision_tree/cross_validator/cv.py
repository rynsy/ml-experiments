#!/usr/bin/python3
"""
    All cross-validation code goes here.
"""

def accuracy(p, a):
    return len(list(filter(lambda x: x[0] == x[1], zip(p, a)))) / len(a)

def misclassification(p, a):
    return 1.0 - accuracy(p, a)
