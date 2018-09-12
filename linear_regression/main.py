#!/usr/bin/python
from data_wrangler import *
from linear import *
from plot import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from time import sleep
#need to compute 0-4th order models for then compute mse for each
# on the training set of data

features_train, labels_train = loadTrainingData()
features_test, labels_test = loadTestingData()

data_train = scale(np.column_stack((features_train, labels_train)))
data_test = scale(np.column_stack((features_test, labels_test)))

## Scale and prep data, train the models.

x1 = data_train[:, :-1]
y = data_train[:, -1]

x0 = np.ones(np.shape(x1))
x2 = x1 ** 2
x3 = x1 ** 3
x4 = x1 ** 4

zeroth_order_gd = fit_lr_gd(x0, y)
first_order_gd = fit_lr_gd(np.column_stack((x0,x1)), y)
second_order_gd = fit_lr_gd(np.column_stack((x0,x1,x2)), y)
third_order_gd = fit_lr_gd(np.column_stack((x0,x1,x2,x3)), y)
fourth_order_gd = fit_lr_gd(np.column_stack((x0,x1,x2,x3,x4)), y)

print "Zeroth order (GD): ", zeroth_order_gd
print "First order (GD): ", first_order_gd
print "Second order (GD): ", second_order_gd
print "Third order (GD): ", third_order_gd
print "Fourth order (GD): ", fourth_order_gd

zeroth_order_nm = fit_lr_normal(x0, y)
first_order_nm = fit_lr_normal(np.column_stack((x0,x1)), y)
second_order_nm = fit_lr_normal(np.column_stack((x0,x1,x2)), y)
third_order_nm = fit_lr_normal(np.column_stack((x0,x1,x2,x3)), y)
fourth_order_nm = fit_lr_normal(np.column_stack((x0,x1,x2,x3,x4)), y)

print "Zeroth order (normal): ", zeroth_order_nm
print "First order (normal): ", first_order_nm
print "Second order (normal): ", second_order_nm
print "Third order (normal): ", third_order_nm
print "Fourth order (normal): ", fourth_order_nm

## Scale and prep testing data
x1 = data_test[:, :-1]
y = data_test[:, -1]

x0 = np.ones(np.shape(x1))
x2 = x1 ** 2
x3 = x1 ** 3
x4 = x1 ** 4

zero_predict_gd = predict_lr(zeroth_order_gd, x0)
first_predict_gd = predict_lr(first_order_gd, np.column_stack((x0,x1)))
second_predict_gd = predict_lr(second_order_gd, np.column_stack((x0,x1,x2)))
third_predict_gd = predict_lr(third_order_gd, np.column_stack((x0,x1,x2,x3)))
fourth_predict_gd = predict_lr(fourth_order_gd, np.column_stack((x0,x1,x2,x3,x4)))

print "MSE (GD) 0: ", compute_mse(y, zero_predict_gd)
print "MSE (GD) 1: ", compute_mse(y, first_predict_gd)
print "MSE (GD) 2: ", compute_mse(y, second_predict_gd)
print "MSE (GD) 3: ", compute_mse(y, third_predict_gd)
print "MSE (GD) 4: ", compute_mse(y, fourth_predict_gd)

zero_predict_nm = predict_lr(zeroth_order_nm, x0)
first_predict_nm = predict_lr(first_order_nm, np.column_stack((x0,x1)))
second_predict_nm = predict_lr(second_order_nm, np.column_stack((x0,x1,x2)))
third_predict_nm = predict_lr(third_order_nm, np.column_stack((x0,x1,x2,x3)))
fourth_predict_nm = predict_lr(fourth_order_nm, np.column_stack((x0,x1,x2,x3,x4)))

print "MSE (NM) 0: ", compute_mse(y, zero_predict_nm)
print "MSE (NM) 1: ", compute_mse(y, first_predict_nm)
print "MSE (NM) 2: ", compute_mse(y, second_predict_nm)
print "MSE (NM) 3: ", compute_mse(y, third_predict_nm)
print "MSE (NM) 4: ", compute_mse(y, fourth_predict_nm)

plotModel(zeroth_order_gd, x0, y)
sleep(2)
plotModel(first_order_gd, x1, y)
sleep(2)
plotModel(second_order_gd, x2, y)
sleep(2)
plotModel(third_order_gd, x3, y)
sleep(2)
plotModel(fourth_order_gd, x4, y)
