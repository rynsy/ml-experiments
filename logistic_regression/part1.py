from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
from sklearn.cross_validation import train_test_split

THRESH = 0.5

def loadSynthetic(n):
    """
        Takes a number, returns a pandas dataframe
    """
    return pd.read_csv("data/synthetic-" + str(n) + ".csv", header=None)

def prepData(d):
    """
        expects pandas dataframe, returns train_features, test_features, train_labels, test_labels
    """
    features = d.iloc[:,0:-1]
    labels = d.iloc[:, -1]
    return train_test_split(features, labels, test_size=0.33, random_state=44)

class LogisticRegression(object):
    """
        learn : learning rate for gradient descent
        iterations: number of iterations for gradient descent
    """
    def __init__(self, learn=0.01, iterations=50):
        self.learn = learn
        self.iterations = iterations

    def fit(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        self.cost = []
        for i in range(self.iterations):
            y_hat = self.activation(X)
            error = (y - y_hat)
            n_gradient = X.T.dot(error) #negative gradient
            self.w[1:] += self.learn * n_gradient
            self.w[0] += self.learn * sum(error)
            self.cost.append(self.logit_cost(y, self.activation(X)))
        return self
    
    def logit_cost(self, y, y_hat):
        return - y.dot(np.log(y_hat)) - ((1 - y).dot(np.log(1 - y_hat)))

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def activation(self, X):
        return self.sigmoid(self.net_input(X))

    def predict_proba(self, X):
        """
            similar to sklearn implementation
        """
        return self.activation(X)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        #return np.where(self.activation(X) >= THRESH, 1, 0)

    def misclassification(self, X, y):
        """
            incorrect predictions / total examples
        """
        res = self.predict(X)
        return sum(map(lambda (x,y): x != y, zip(y,res))) / len(y)

def generateFigures(expansion_func, graphName="part1"):
    graphs = plt.figure()
    for i in range(1,7):
        data = loadSynthetic(i)
        X,Y = data.iloc[:,0:2], data.iloc[:,-1]
        colorize = lambda d: map(lambda x: "r" if x == 1 else "b", d) #only need two labels/colors
        h = 0.2 #mesh stepsize
        
        clf = LogisticRegression() #todo changeme
        clf = clf.fit(expansion_func(X.iloc[:,0],X.iloc[:,1]), Y)
        
        x_min, x_max = X.iloc[:,0].min() - 1, X.iloc[:,0].max() + 1
        y_min, y_max = X.iloc[:,1].min() - 1, X.iloc[:,1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict_proba(expansion_func(xx,yy)) #todo changeme
        Z = Z.reshape(xx.shape)
    
        fig = graphs.add_subplot(2,3,i)
        fig.set_xlabel("X1")
        fig.set_ylabel("X2")
        fig.set_title("synthetic-" + str(i))
        plt.set_cmap('prism')
        cs = plt.imshow(Z,
                        extent=(x_min,x_max,y_max,y_min),
                        cmap=plt.cm.jet)
        plt.clim(0,1)
        levels = np.array([.5])
        cs_line = plt.contour(xx,yy,Z,levels)
        plt.scatter(X.iloc[:,0], X.iloc[:,1], c=colorize(Y), edgecolors='k', cmap=plt.cm.jet)
    CB = plt.colorbar(cs)
    plt.savefig("figures/" + str(graphName) + ".png", dpi=300)
    plt.show()

def generateMisclassificationRates():
    for i in range(1,7):
        data = loadSynthetic(i)
        X,Y = data.iloc[:,0:2], data.iloc[:,-1]
        train_features, test_features, train_labels, test_labels = prepData(data)
        clf = LogisticRegression()
        clf = clf.fit(expansion_func(train_features.iloc[:,0], train_features.iloc[:,1]), train_labels)
        print "Misclassification for Synthetic-" + str(i) + ": " + str(clf.misclassification(test_features, test_labels))

expand = lambda x,y: np.stack((np.ones(x.size), # add the bias term    
                                x.ravel(), # make the matrix into a vector
                                y.ravel(),    
                                np.array([m * n for (m,n) in zip(x.ravel(),y.ravel())]),
                                x.ravel()**2,    
                                y.ravel()**2),
                                axis=1) # add a quadratic term for fun

unravel = lambda x,y: np.c_[x.ravel(), y.ravel()]

generateFigures(expansion_func=unravel, graphName="No Expansion")
generateFigures(expansion_func=expand, graphName="Expanded")

expandXY = lambda x,y: np.stack((np.ones(x.size), # add the bias term    
                                np.array([m * n for (m,n) in zip(x.ravel(),y.ravel())]),
                                x.ravel()**2,    
                                y.ravel()**2),
                                axis=1)

generateFigures(expansion_func=expandXY, graphName="XY")


