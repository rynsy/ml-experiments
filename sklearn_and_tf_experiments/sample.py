from __future__ import division
import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

PRINT_MATRIX = False

def true_positives(m):
    s = 0
    t = 0
    for i in range(len(m)):
        s += m[i][i]
        t += sum(m[i])
    return s / t 

def random_confusion_matrix(labels):
    pass

data_train = np.genfromtxt('data/caltechTrainData.dat')
data_train = data_train / 255

data_train_labels = np.genfromtxt('data/caltechTrainLabel.dat')

x_train, x_test, y_train, y_test = train_test_split(data_train, 
        data_train_labels, test_size=0.3, random_state=55)

clf = SVC(decision_function_shape='ovo', gamma=0.001)
clf = clf.fit(x_train, y_train)
print("SVC Score: ", cross_val_score(clf, x_test, y_test).mean())
confus = metrics.confusion_matrix(y_test, clf.predict(x_test))
if PRINT_MATRIX:
    print("SVC True Positive Rate (training data): ", true_positives(confus))
    print("SVC Confusion Matrix: ", confus)

clf = LogisticRegression()
clf = clf.fit(x_train, y_train)
print("LogReg Score: ", cross_val_score(clf, x_test, y_test).mean())
confus = metrics.confusion_matrix(y_test, clf.predict(x_test))
if PRINT_MATRIX:
    print("LogReg True Positive Rate (training data): ", true_positives(confus))
    print("LogReg Confusion Matrix: ", confus)

"""
clf = LinearSVC()
clf = clf.fit(x_train, y_train)
print("LinearSVC Score: ", clf.score(x_test, y_test))
confus = metrics.confusion_matrix(y_test, clf.predict(x_test))
if PRINT_MATRIX:
    print("LinearSVC True Positive Rate (training data): ", true_positives(confus))
    print("LinearSVC Confusion Matrix: ", confus)

clf = RandomForestClassifier()
clf = clf.fit(x_train, y_train)
print("Random Forest Score: ", clf.score(x_test, y_test))
confus = metrics.confusion_matrix(y_test, clf.predict(x_test))
if PRINT_MATRIX:
    print("Random Forest True Positive Rate (training data): ", true_positives(confus))
    print("Random Forest Confusion Matrix: ", confus)

clf = KNeighborsClassifier()
clf = clf.fit(x_train, y_train)
print("KNN Score: ", clf.score(x_test, y_test))
confus = metrics.confusion_matrix(y_test, clf.predict(x_test))
if PRINT_MATRIX:
    print("KNN True Positive Rate (training data): ", true_positives(confus))
    print("KNN Confusion Matrix: ", confus)
"""
