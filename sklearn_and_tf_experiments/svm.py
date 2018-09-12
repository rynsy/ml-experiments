from __future__ import division
import numpy as np
from sklearn import grid_search
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.svm import SVC
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

parameters = {'kernel': ['linear', 'rbf'], 'C':[1,10,100,1000,10000],
		'gamma': [0.01, 0.001, 0.0001, 0.00001]}

clf = grid_search.GridSearchCV(SVC(), parameters).fit(x_train, y_train)
clf = clf.best_estimator_

print("SVC Score: ", cross_val_score(clf, x_test, y_test).mean())
confus = metrics.confusion_matrix(y_test, clf.predict(x_test))
if PRINT_MATRIX:
    print("SVC True Positive Rate (training data): ", true_positives(confus))
    print("SVC Confusion Matrix: ", confus)

