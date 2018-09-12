from __future__ import division
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

PRINT_MATRIX = False

data_train = np.genfromtxt('data/caltechTrainData.dat')
data_train = data_train / 255

data_train_labels = np.genfromtxt('data/caltechTrainLabel.dat')

x_train, x_test, y_train, y_test = train_test_split(data_train, 
        data_train_labels, test_size=0.3, random_state=55)

#clf = MLPClassifier(learning_rate='adaptive', solver='lbfgs')
clf = MLPClassifier()
clf = clf.fit(x_train, y_train)
print("MLP Score: ", cross_val_score(clf, x_test, y_test).mean())
confus = metrics.confusion_matrix(y_test, clf.predict(x_test))
print("MLP True Positive Rate (training data): ", clf.score(x_test, y_test))
print("MLP Confusion Matrix: ", confus)

