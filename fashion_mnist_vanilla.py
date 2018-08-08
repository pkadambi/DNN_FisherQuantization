import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
# from wt_ops import next_batch
from wt_ops import tf_hard_quantize, sigmoid_quantize, numpy_quantize, numpy_sigmoid_quantize, softmax
from tensorflow.python.ops import gradients_impl
import matplotlib.pyplot as plt
import numpy as np
import time
from train_data_ops import get_fashionMNIST
from train_data_ops import next_batch
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#Define Weight Variables


#Define Quantized Weight variables


# Input layer
x  = tf.placeholder(tf.float32, shape=(None, 784), name='inputs')
y_ = tf.placeholder(tf.float32, shape=(None, 10),  name='labels')

x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolutional layer 1

W_conv1 = tf.get_variable('wconv1',shape= [5, 5, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
b_conv1 = tf.get_variable('bconv1',shape= (32), initializer = tf.random_normal_initializer(stddev=.1))

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer 2
W_conv2 = tf.get_variable('wconv2',shape= (5, 5, 32, 64), initializer=tf.contrib.layers.xavier_initializer())
b_conv2 = tf.get_variable('bconv2',shape= (64), initializer = tf.random_normal_initializer(stddev=.1))

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# Fully connected layer 1
W_fc1 = tf.get_variable('wfc1',shape= (7 * 7 * 64, 1024), initializer=tf.contrib.layers.xavier_initializer())


b_fc1 = tf.get_variable('bfc1',shape= (1024), initializer = tf.random_normal_initializer(stddev=.1))

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_bn = tf.layers.batch_normalization(h_fc1)
# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1_bn , keep_prob)

# Fully connected layer 2 (Output layer)

W_fc2 = tf.get_variable('wfc2',shape= (1024, 10), initializer=tf.contrib.layers.xavier_initializer())
b_fc2 = tf.get_variable('bfc2',shape= (10), initializer = tf.random_normal_initializer(stddev=.1))

softmax_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y = tf.stop_gradient(tf.nn.softmax(softmax_logits))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits=softmax_logits))
# Evaluation functions
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.stop_gradient(tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy'))



#Define, weight list, quantized weights, and regularizer here
optimizer = tf.train.AdamOptimizer(1e-3)
trainable_weights = [W_conv1, b_conv1,  W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]

qlevels = [-1., 1.]
qregions = [0.]
# qwconv1 = tf.stop_gradient(tf_hard_quantize(W_conv1))
# qwconv2 = tf.stop_gradient(tf_hard_quantize(W_conv2))
# qwfc1 = tf.stop_gradient(tf_hard_quantize(W_fc1))
# qwfc2 = tf.stop_gradient(tf_hard_quantize(W_fc2))
regularize_list = [False, False, False, False, False, False, False, False]
is_not_bias = [True, False, True, False, True, False, True, False]
regularized_weights = []
qweight_list = []

for indicator,layer_wts in zip(regularize_list,trainable_weights):
    qweight_list.append(tf.stop_gradient(tf_hard_quantize(layer_wts , qlevels, qregions)))
    if indicator:
        regularized_weights.append(layer_wts)

print('List of weights to regularize')
print(regularized_weights)

print('List of quantized weights')
print(qweight_list)

perturbation_list = []



train_op = optimizer.minimize(loss)

tf.summary.histogram('Weights/Unquantized_wfc1', trainable_weights[0])
tf.summary.histogram('Weights/Unquantized_wfc2', trainable_weights[1])

# tf.summary.histogram('Weights/Quantized_wconv1', qweight_list[0])
# tf.summary.histogram('Weights/Quantized_wconv2', qweight_list[2])
# tf.summary.histogram('Weights/Quantized_wfc1', qweight_list[4])
# tf.summary.histogram('Weights/Quantized_wfc2', qweight_list[6])
tf.summary.histogram('Weights/Quantized_wfc1', qweight_list[0])
tf.summary.histogram('Weights/Quantized_wfc2', qweight_list[1])


# tf.summary.histogram('Gradients/cegrad_wconv1', ce_grads[0])
# tf.summary.histogram('Gradients/cegrad_wconv2', ce_grads[2])
# tf.summary.histogram('Gradients/cegrad_wfc1', ce_grads[4])
# tf.summary.histogram('Gradients/cegrad_wfc2', ce_grads[6])

# tf.summary.histogram('Gradients/cegrad_wfc1', ce_grads[4])
# tf.summary.histogram('Gradients/cegrad_wfc2', ce_grads[6])

# tf.summary.histogram('Gradients/tot_regularizer_grad_wconv1', hessian_vector_product[0])
# tf.summary.histogram('Gradients/tot_regularizer_grad_wconv2', hessian_vector_product[1])
# tf.summary.histogram('Gradients/tot_regularizer_grad_wfc1', hessian_vector_product[2])
# tf.summary.histogram('Gradients/tot_regularizer_grad_wfc2', hessian_vector_product[3])
# tf.summary.histogram('Gradients/tot_regularizer_grad_wfc1', hessian_vector_product[0])
# tf.summary.histogram('Gradients/tot_regularizer_grad_wfc2', hessian_vector_product[1])
# tf.summary.histogram('Gradients/diagonal_component_reg_wconv1', layer_diag_load_amt[0])
# tf.summary.histogram('Gradients/diagonal_component_reg_wconv2', layer_diag_load_amt[1])
# tf.summary.histogram('Gradients/diagonal_component_reg_wfc1', layer_diag_load_amt[2])
# tf.summary.histogram('Gradients/diagonal_component_reg_wfc2', layer_diag_load_amt[3])

# tf.summary.histogram('Gradients/diagonal_component_reg_wfc1', layer_diag_load_amt[0])
# tf.summary.histogram('Gradients/diagonal_component_reg_wfc2', layer_diag_load_amt[1])

tf.summary.scalar('loss_and_acc/ce_loss', loss)
tf.summary.scalar('loss_and_acc/accuracy', accuracy)
# tf.summary.scalar('loss_and_acc/regularizer_constant', gamma)

sess = tf.Session()
summary_writer = tf.summary.FileWriter('./logs/fashion_vanilla', sess.graph)
summary_op = tf.summary.merge_all()
lossval=[]
accval=[]
#Training loop
sess.run(tf.global_variables_initializer())
n_iters=6000
REGULARIZER_GAMMA = 0.0

elapsed = 0.0
dataset = get_fashionMNIST()

x_train = dataset[0]
y_train = dataset[1]

x_cv = dataset[2]
y_cv = dataset[3]

x_test = dataset[4]
y_test = dataset[5]
start = time.time()
for i in range(n_iters):

    batch = next_batch( x_train, y_train, 128)
    # print(np.shape(batch[0]))
    # print(np.shape(batch[1]))
    # exit()
    if i % 100 == 0:
        # summ, train_accuracy = sess.run([summary_op, accuracy],feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0, gamma: regularizer_constant })
        # print('Step %d, training accuracy %g, elapsed time %1.1f' % (i, train_accuracy, time.time() - start))
        # summary_writer.add_summary(summ, i)

        train_accuracy = sess.run([ accuracy],feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0,})
        print('Step %d, training accuracy %g, elapsed time %1.1f' % (i, train_accuracy[0], time.time() - start))

    # summ, _, l,acc =  sess.run([summary_op, train_op, loss, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, })
    summ, _ =  sess.run([summary_op, train_op], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, })
    summary_writer.add_summary(summ, i)
    # lossval.append(l)
    # accval.append(acc)
'''
Unquantized Test accuracy
'''
# print('UNQUANTIZED Validation accuracy %g' % sess.run(accuracy, feed_dict={
#     x: x_cv, y_: y_cv, keep_prob: 1.0}))
# print('UNQUANTIZED Test accuracy %g' % sess.run(accuracy, feed_dict={
#     x: x_test, y_: y_test, keep_prob: 1.0}))
'''
Quantize regularized layers (WEIGHTS ONLY) and get CV acc
'''
quantize_list = [False, False, False, False, True, False, True, False]

for indicator, layerwt, qlayerwt in zip (quantize_list, trainable_weights, qweight_list):
    if indicator:
        sess.run(tf.assign(layerwt,qlayerwt))
        print(qlayerwt)
print('FC Layers QUANTIZED CV accuracy %g' % sess.run(accuracy, feed_dict={
    x: x_cv, y_: y_cv, keep_prob: 1.0}))
print('FC Layers QUANTIZED Test accuracy %g' % sess.run(accuracy, feed_dict={
    x: x_test, y_: y_test, keep_prob: 1.0}))

'''
Quantize ALL layers (WEIGHTS ONLY) and get acc
'''
# for indicator, layerwt, qlayerwt in zip (is_not_bias, trainable_weights, qweight_list):
#     if indicator:
#         sess.run(tf.assign(layerwt,qlayerwt))
#         print(qlayerwt)
# print('FULLY QUANTIZED Validation accuracy %g' % sess.run(accuracy, feed_dict={
#     x: x_cv, y_: y_cv, keep_prob: 1.0}))
# print('FULLY QUANTIZED Test accuracy %g' % sess.run(accuracy, feed_dict={
#     x: x_test, y_: y_test, keep_prob: 1.0}))



#Plot in matplotlib
# t = np.arange(0,n_iters).reshape(n_iters,1)+1
# lossval = np.array(lossval).reshape(n_iters,1)
# accval = np.array(accval).reshape(n_iters,1)

# plt.figure(1)
# plt.subplot(2,1,1)
# plt.plot(t, lossval, 'g', label = 'Loss')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')

# plt.subplot(2,1,2)
# plt.plot(t, accval, 'b', label = 'Accuracy')
# plt.xlabel('Iteration')
# plt.ylabel('Acc')


plt.show()