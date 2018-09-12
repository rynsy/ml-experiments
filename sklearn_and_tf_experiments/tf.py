from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics

def true_positives(m):
    s = 0
    t = 0
    for i in range(len(m)):
        s += m[i][i]
        t += sum(m[i])
    return s / t 

def next_batch():
    #should be similar to the mnist batch function in siraj's video
    pass

data_train = np.genfromtxt('data/caltechTrainData.dat')
data_train = data_train / 255

data_train_labels = np.genfromtxt('data/caltechTrainLabel.dat')

x_train, x_test, y_train, y_test = train_test_split(data_train, 
        data_train_labels, test_size=0.3, random_state=55)

images_placeholder = tf.placeholder(tf.float32,[None, 2700])#not sure if these should match the 30x30 shape of the actual image, or the 1x2700 shape of the data
labels_placeholder = tf.placeholder(tf.int32, [None, 18]) #Should probably be image_num x 1

with tf.name_scope("Wx_b") as scope:
    model = tf.nn.softmax(tf.matmul(x, W) + b)

w_h = tf.histogram_summary("weights", W)
b_h = tf.histogram_summary("biases", b)

with tf.name_scope("cost_function") as scope:
    # minimize error using cross_entropy
    cost_function = -tf.reduce_sum(y*tf.log(model))
    tf.scalar_summary("cost_function", cost_function)

with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.initialize_all_variables()

merged_summary_op = tf.merge_all_summaries()

with tf.Session() as sess:
    sess.run(init)
    
    #log
    summary_writer = tf.train.SummaryWriter('', graph_dev = sess.graph_dev)
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int( total / batch_size) #fixme
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch() #fixme
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_sumary(summary_str, iteration*total_batch + i)

        if iteration % display_step == 0:
            print("Iteration:", '%04d' %(iteration+1), "cost=", "{:9f}".format(avg_cost))
    print("Tuning completed!")

    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y,y))

    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy: ", accuracy.eval({x: x_test, y: y_test})
