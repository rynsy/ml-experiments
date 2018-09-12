#!/usr/bin/python
from sklearn.linear_model import LinearRegression
from data_wrangler import *
import matplotlib.pyplot as plt
from sklearn import preprocessing

features_train, labels_train = loadTrainingData()
features_test, labels_test = loadTestingData()
train_color = "b"
test_color = "r"

reg = LinearRegression()        # Sklearn answer
reg.fit(features_train, labels_train)

for feature, target in zip(features_test, labels_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(features_train, labels_train):
    plt.scatter( feature, target, color=train_color ) 

plt.scatter(features_test[0], labels_test[0], color=test_color, label="test")
plt.scatter(features_test[0], labels_test[0], color=train_color, label="train")

plt.plot( features_test, reg.predict(features_test) )
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()
