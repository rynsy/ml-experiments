#!/usr/bin/python3
from math import *

def divide(rows, column, value):
    """
        divides data into two different sets, one where the value in row[column]
        is greater than the provided value, and one where the value of row[column]
        is lower.
    """
    if isinstance(rows[0][column], str) or isinstance(rows[0][column], bool): # for categorical spliting
        splitter = lambda row: row[column] == value 
    else:                                # for splitting on numerical value
        splitter = lambda row: row[column] >= value
    set1 = [row for row in rows if splitter(row)]
    set2 = [row for row in rows if not splitter(row)]
    return (set1, set2)

def unique(rows):
    """
        returns a dictionary of counts of unique labels in the dataset
        Looks at last column, assumes this is the label for the record. 
    """
    results = {}
    for row in rows:
        r = row[len(row)-1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results

def entropy(rows):
    """
        returns the amount of entropy for each element across the entire set.
    """
    classes = unique(rows)
    ent = 0.0
    for c in classes.keys():
        e = float(classes[c]/len(rows))
        ent = ent - e * (log(e) / log(2))
    return ent

def infogain(current, set1, set2):
    """
        returns the amount of information gained by a split. 
        Where:
            current: 
                current entropy value
            set1:
                set where split is true for the splitting criteria
            set2:
                set where split is false for the splitting criteria
    """
    p = len(set1) / (len(set1) + len(set2))
    return current - p * entropy(set1) - (1-p) * entropy(set2)

def gini(rows):
    """
        returns the gini impurity of the provided dataset. 
        returns 1 - sum(probability of each label squared)
    """
    classes = unique(rows)
    imp = 0.0
    for c in classes.keys():
        e = float(classes[c]/len(rows))
        imp = imp + e ** 2
    return 1 - imp
