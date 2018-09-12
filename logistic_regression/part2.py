from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

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

graphs = plt.figure()

for i in range(1,7):
    data = loadSynthetic(i)
    X,Y = data.iloc[:,0:2], data.iloc[:,-1]
    train_features, test_features, train_labels, test_labels = prepData(data)
    colorize = lambda d: map(lambda x: "r" if x == 1 else "b", d) #only need two labels/colors
    h = 0.2 #mesh stepsize

    clf = LogisticRegression()
    clf = clf.fit(train_features, train_labels)

    x_min, x_max = X.iloc[:,0].min() - 1, X.iloc[:,0].max() + 1
    y_min, y_max = X.iloc[:,1].min() - 1, X.iloc[:,1].max() + 1


    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
    #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])[
    Z = Z.reshape(xx.shape)
    
    fig = graphs.add_subplot(2,3,i)
    fig.set_xlabel("X1")
    fig.set_ylabel("X2")
    fig.set_title("synthetic-" + str(i))
    #plt.pcolormesh(xx, yy, Z, cmap=plt.cm.jet)
    plt.set_cmap('prism')
    cs = plt.imshow(Z,
                    extent=(x_min,x_max,y_max,y_min),
                    cmap=plt.cm.jet)
    plt.clim(0,1)
    levels = np.array([.5])
    cs_line = plt.contour(xx,yy,Z,levels)
    plt.scatter(X.iloc[:,0], X.iloc[:,1], c=colorize(Y), edgecolors='k', cmap=plt.cm.jet)

CB = plt.colorbar(cs)
plt.savefig("figures/part2.png", dpi=300)
plt.show()
