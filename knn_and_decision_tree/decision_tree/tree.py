#!/usr/bin/python3
from decision_tree import *
import numpy as np
from operator import itemgetter

class node:
    """
        Single node used for the decision tree. 
        col:
            index of the feature column the node represents
        value:
            value for that feature column
        results:
            None if node has children, dictionary of unique labels if not.
        tb:
            branch where splitting condition is true.
        fb: 
            branch where splitting condition is false.
    """
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb            #true branch
        self.fb = fb            #false branch

def getColumnValues(approach, rows, col):
    """
        returns the values of the current column in a format expected
        by constructTree. This allows you to change the splits as a 
        parameter to constructTree.

        set interval_points to the number of points needed if you are
        splitting over an interval.
    """
    interval_points = 100

    if approach == 'unique':
        col_vals = {}
        for row in rows:
            col_vals[row[col]] = 1
        return col_vals
    elif approach == 'interval':
        row_array = np.array(rows)
        low = row_array.min(axis=0)[col]
        high = row_array.max(axis=0)[col]
        return dict(enumerate(np.linspace(low, high, interval_points)))
    else:
        raise Exception("unknown split function")

def constructTree(rows, splitter=entropy, split_on='unique', depth=0, max_depth=3):
    """
        Builds a decision tree of nodes. Splits the dataset using each
        unique value in the column instead of an interval. uses infogain
        to determine whether or not a feature is good for splitting.
        
        rows:
            an array of features. Expects a different format from other 
            functions. Data should be a 2D array where the last column
            is the labels for the data. 

        splitter:
            parameter for splitting data into groups. A function that
            should return a numerical value. Defaults to entropy.
            Set this parameter to 'gini' to use the gini impurity 
            metric.
        split_on:
            used to change the splits of the data. There are two options
            as of this writing: interval (split on a interval coded into 
            getColumnValues), or unique (split on every value for the feature)

        depth:
            used for tracking the depth of the tree. Algorithm returns if
            depth == max_depth. defaults to 0, 
    """
    if len(rows) == 0: return node()
    if depth == max_depth: return node(results=unique(rows))

    current_score = splitter(rows)

    best_gain = 0.0
    best_criteria = None
    best_sets = None

    col_count = len(rows[0]) - 1
    for col in range(0, col_count):
        #make list of all possible values to split on
        col_vals = getColumnValues(split_on, rows, col)
        for value in col_vals.keys():
            (set1, set2) = divide(rows, col, value)
            gain = infogain(current_score, set1, set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    if best_gain > 0:
        trueBranch = constructTree(best_sets[0], depth=depth + 1)
        falseBranch = constructTree(best_sets[1], depth=depth + 1)
        return node(col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch)
    else:
        return node(results=unique(rows))

def printTree(tree, indent=''):
    """
        Builds a text representation of the tree. Useful for
        debugging and viewing what the splitting features are.
    """
    if tree.results != None:
        print(str(tree.results))
    else:
        print(str(tree.col) + ' : >=' + str(tree.value) + ' ? ')
        print(indent + 'True -> ', end=" ")
        printTree(tree.tb, indent+'  ')
        print(indent + 'False ->', end=" ")
        printTree(tree.fb, indent+'  ')

def getResultsDict(row, tree):
    """
        if the resulting node isn't a leaf, it will contain a dictionary of 
        each label, and the number of rows with that label for the nodes
        children. returns dictionary formatted like this:
            {label1: label_count, label2: label_count}

        if the resulting node is a leaf, it will return a dictionary of the
        predicted label and the number of rows that have that label and can 
        be categorized with the path to this leaf. 
        returns dictionary formatted like this:
            {label: label_count}
    """
    if tree.results != None:
        return tree.results
    else:
        value = row[tree.col]
        if value >= tree.value: branch = tree.tb
        else: branch = tree.fb
    return getResultsDict(row, branch)

def classify(row, tree):
    """
        retrieves the label from the dictionary you get from crawling the tree.
        expects one row, returns a label.
    """
    r = getResultsDict(row, tree)
    return max(r.items(), key=itemgetter(1))[0]

def predict(rows, tree):
    """
        retrieves labels for a group of rows. Used for replicating sklearn functions
        like "tree.predict"
    """
    labels = []
    for row in rows:
        labels.append(classify(row, tree))
    return labels
