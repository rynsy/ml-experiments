import numpy as np
import matplotlib
matplotlib.use('Agg') #used for working remotely without x-windows
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split


data_train = np.genfromtxt('data/caltechTrainData.dat')
data_train = data_train / 255
data_train_labels = np.genfromtxt('data/caltechTrainLabel.dat')

x_train, x_test, y_train, y_test = train_test_split(data_train, data_train_labels, \
                                        test_size=0.3, random_state = 42)

clf = RandomForestClassifier()
clf.fit(x_train, y_train)

print("Score: ", clf.score(x_test, y_test))

#im_train = data_train[0,:].reshape((30,30,3), order='F')
#plt.imshow(im_train)
#plt.savefig("something.png")
#plt.show()
