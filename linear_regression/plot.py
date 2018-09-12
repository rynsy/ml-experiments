#!/usr/bin/python
import matplotlib.pyplot as plt
from linear import *

def plotModel(model, features, labels):
    """
        Where model is a regression model with the member functions

        fit(features, labels) => None
            and
        predict(features) => labels

    """
    test_color = "r"
    for feature, target in zip(features, labels):
        plt.scatter( feature, target, color=test_color ) 
    
    plt.plot(features, predict_lr(model, features), color='blue', linewidth=3)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def plotUpdatingModel(model):
    """
        expects the model to be a generator expression that will 
        yield theta for the plot.

        used for debugging.
    """

    features_train, labels_train = loadTrainingData()
    features_test, labels_test = loadTestingData()
    train_color = "g"
    test_color = "r"

    plt.ion()

    for feature, target in zip(features_test, labels_test):
        plt.scatter( feature, target, color=test_color ) 
    for feature, target in zip(features_train, labels_train):
        plt.scatter( feature, target, color=train_color ) 

    plt.scatter(features_test[0], labels_test[0], color=test_color, label="test")
    plt.scatter(features_test[0], labels_test[0], color=train_color, label="train")

    graph = plt.plot( features_test, predict(next(model), features_test) )[0]
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    #plt.show()
    while True:
        graph.set_ydata(predict_lr(next(model), features_test))
        plt.draw()
        plt.pause(0.01)
