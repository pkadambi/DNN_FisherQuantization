import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from wt_ops import tf_hard_quantize, sigmoid_quantize, numpy_quantize, numpy_sigmoid_quantize, softmax
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import gradients_impl

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

x = tf.placeholder(tf.float32, shape = (None, 784), name='Inputs')
y = tf.placeholder(tf.float32, shape = (None, 10), name='Labels')
gamma = tf.placeholder(tf.float32, shape = (), name='reg_constant')
nwts = 7840
# wts = tf.get_variable('Weights',shape= (784,10), initializer = tf.random_normal_initializer(stddev=.001))
w = tf.get_variable(name='w', shape=[784, 10], initializer=tf.contrib.layers.xavier_initializer())
bias = tf.get_variable('bias',shape= (10), initializer = tf.random_normal_initializer(stddev=.1))


# 0.1000    0.1292    0.1668    0.2154    0.2783    0.3594    0.4642    0.5995    0.7743    1.0000

w_pert = tf.random_normal(shape=tf.shape(w), mean=0., stddev=.0001)
perturbation = tf.stop_gradient(w - w_pert)


logits = tf.matmul(x, w) + bias
y_ = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.stop_gradient(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits))


optimizer = tf.train.AdamOptimizer()
ce_grads = tf.gradients(loss, [w, bias])
ce_grads_w = ce_grads[0]
hvp = gradients_impl._hessian_vector_product(loss, [w], [perturbation])
diag_load_amt = gamma * .001 * perturbation
reg_grad = gamma * 2.0 * hvp + diag_load_amt
reg_grad = tf.reshape(reg_grad , tf.shape(w))

tot_grads = ce_grads_w + reg_grad

tf.summary.histogram('weights', w)
tf.summary.histogram('regularizer_gradient', reg_grad)
tf.summary.histogram('ce_gradient', ce_grads_w)

train_op = optimizer.apply_gradients(zip([tot_grads, ce_grads[1]], [w, bias]))

n_iters = 5000
batch_size = 256
n_fisher_iters=1000
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)

sess = tf.Session()

summary_writer = tf.summary.FileWriter('./logs/linear_mdl', sess.graph)
summary_op = tf.summary.merge_all()

lossval=[]
accval=[]
sess.run(tf.global_variables_initializer())

regularizer_const=0.
for i in range(0, n_iters):
    x_batch, y_batch = mnist.train.next_batch(batch_size)

    if i<(n_iters-n_fisher_iters):
        regularizer_const=0.
    else:
        regularizer_const=10.

    # _, l, acc, w_, b_ = sess.run([train_op, loss, accuracy, w, bias], feed_dict={x: x_batch, y: y_batch, gamma:regularizer_const})
    summ, _, l, acc, w_, b_ = sess.run([summary_op, train_op, loss, accuracy, w, bias], feed_dict={x: x_batch, y: y_batch, gamma:regularizer_const})
    summary_writer.add_summary(summ, i)
    lossval.append(l)
    accval.append(acc)

    if i%50:
        print('\nIteration: '+str(i)+'\nAccuracy: '+str(acc)+'\nLoss: '+str(l)+'\n')

print('UNPERTURBED Test accuracy %g' % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
sess.run(tf.assign(w, w_pert))

print('PRETURBED test accuracy %g' % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))














