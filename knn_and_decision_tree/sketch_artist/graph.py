#!/usr/bin/python3
"""
    All code for displaying/visualizing data goes here.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from decision_tree import *
from cross_validator import *
from data_wrangler import *
from knn import *
from sklearn import neighbors as n
import sklearn.tree as t
POINT_NUM = 100

def calculateAndPlotDT():
    global POINT_NUM
    plot_colors = "br"
    for pindx in range(1,5):
        features, labels = loadSyntheticData(pindx)
        n_classes = len(list(set(labels)))
        X = preprocessing.scale(features)
        idx = np.arange(X.shape[0])
        np.random.seed(42)
        np.random.shuffle(idx)
        X = X[idx]
        y = labels[idx]
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        myTree = constructTree(np.column_stack((X,y)))
        labels_pred = predict(np.column_stack((X,y)), myTree)
        m_err = misclassification(labels_pred, y)
        print("For dataset synthetic-%d, the model misclassified %f of labels." % (pindx, m_err))
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, POINT_NUM), np.linspace(y_min, y_max, POINT_NUM))
        z = np.array(predict(np.c_[xx.ravel(), yy.ravel()], myTree))
        z = z.reshape(xx.shape)
        ax = plt.subplot(1,4,pindx)
        ax.title.set_text("synthetic-"+str(pindx))
        cs = plt.contourf(xx,yy,z, cmap=plt.cm.Paired)
        plt.xlabel("x0")
        plt.ylabel("x1")
        plt.axis("tight")
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)
    plt.legend()
    plt.savefig("graphs/decision_tree_boundary.png")
    plt.show()

def plotKNN(optimal_k):
    global POINT_NUM
    plot_colors = "br"
    for pindx in range(1,5):
        features, labels = loadSyntheticData(pindx)
        n_classes = len(list(set(labels)))
        X = preprocessing.scale(features)
        idx = np.arange(X.shape[0])
        np.random.seed(42)
        np.random.shuffle(idx)
        X = X[idx]
        y = labels[idx]
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, POINT_NUM), np.linspace(y_min, y_max, POINT_NUM))
        z = np.array(kNN(np.c_[xx.ravel(), yy.ravel()], optimal_k[pindx-1], X, y))
        z = z.reshape(xx.shape)
        ax = plt.subplot(1,4,pindx)
        ax.title.set_text("synthetic-"+str(pindx))
        cs = plt.contourf(xx,yy,z, cmap=plt.cm.Paired)
        plt.xlabel("x0")
        plt.ylabel("x1")
        plt.axis("tight")
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)
    plt.legend()
    plt.savefig("graphs/knn_boundary.png")
    plt.show()

def plotLibraryKNN(optimal_k):
    global POINT_NUM
    plot_colors = "br"
    for pindx in range(1,5):
        features, labels = loadSyntheticData(pindx)
        n_classes = len(list(set(labels)))
        X = preprocessing.scale(features)
        idx = np.arange(X.shape[0])
        np.random.seed(42)
        np.random.shuffle(idx)
        X = X[idx]
        y = labels[idx]
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, POINT_NUM), np.linspace(y_min, y_max, POINT_NUM))
        neighbs = n.KNeighborsClassifier(n_neighbors=optimal_k[pindx - 1])
        neighbs.fit(X,y)
        z = np.array(neighbs.predict(np.c_[xx.ravel(), yy.ravel()]))
        z = z.reshape(xx.shape)
        ax = plt.subplot(1,4,pindx)
        ax.title.set_text("synthetic-"+str(pindx))
        cs = plt.contourf(xx,yy,z, cmap=plt.cm.Paired)
        plt.xlabel("x0")
        plt.ylabel("x1")
        plt.axis("tight")
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)
    plt.legend()
    plt.savefig("graphs/sklearn_knn_boundary.png")
    plt.show()

def plotLibraryDT():
    global POINT_NUM
    plot_colors = "br"
    for pindx in range(1,5):
        features, labels = loadSyntheticData(pindx)
        n_classes = len(list(set(labels)))
        X = preprocessing.scale(features)
        idx = np.arange(X.shape[0])
        np.random.seed(42)
        np.random.shuffle(idx)
        X = X[idx]
        y = labels[idx]
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        tree = t.DecisionTreeClassifier(max_depth=3)
        tree = tree.fit(X,y)
        labels_pred = tree.predict(X)
        m_err = misclassification(labels_pred, y)
        print("For dataset synthetic-%d, sklearn's model misclassified %f of labels." % (pindx, m_err))
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, POINT_NUM), np.linspace(y_min, y_max, POINT_NUM))
        z = np.array(tree.predict(np.c_[xx.ravel(), yy.ravel()]))
        z = z.reshape(xx.shape)
        ax = plt.subplot(1,4,pindx)
        ax.title.set_text("synthetic-"+str(pindx))
        cs = plt.contourf(xx,yy,z, cmap=plt.cm.Paired)
        plt.xlabel("x0")
        plt.ylabel("x1")
        plt.axis("tight")
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)
    plt.legend()
    plt.savefig("graphs/sklearn_decision_tree_boundary.png")
    plt.show()
