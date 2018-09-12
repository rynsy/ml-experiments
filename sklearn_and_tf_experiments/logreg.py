import numpy as np
import matplotlib
matplotlib.use('Agg') #used for working remotely without x-windows
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics

data_train = np.genfromtxt('data/caltechTrainData.dat')
data_train = data_train / 255
data_train_labels = np.genfromtxt('data/caltechTrainLabel.dat')

x_train, x_test, y_train, y_test = train_test_split(data_train, data_train_labels, \
                                        test_size=0.3, random_state = 42)

parameters = {'penalty':['l2'], 'C':[1.0, 0.5, 0.1], 
		'solver':['newton-cg', 'lbfgs' ],
		'multi_class':['ovr','multinomial']}

clf = GridSearchCV(LogisticRegression(), parameters).fit(x_train, y_train)
clf = clf.best_estimator_
f = open("log_reg_output.txt", "w")

f.write("LogReg score: " + str(cross_val_score(clf, x_test, y_test).mean()) + "\n")
confus = metrics.confusion_matrix(y_test, clf.predict(x_test))
f.write("LogReg True Positive Rate (training data): " + str(clf.score(x_test, y_test)) + "\n")
f.write("LogReg Confusion Matrix: " + str(confus) + "\n")
f.close()

data_test = np.genfromtxt('data/caltechTestData.dat')
data_test = data_test / 255

f = open("data/caltechTestLabel.dat", "w")
predicted_labels = clf.predict(data_test)
for l in predicted_labels:
    f.write(str(l) + "\n")
f.close()
