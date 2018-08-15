from tensorflow.python.ops import gradients_impl
import tensorflow as tf
import numpy as np
import time



# return the value after hard sigmoid
def hard_sigmoid(x):
    return tf.contrib.keras.backend.clip((x+1.0)/2.0, 0.0, 1.0)


    # binarize the weight
def binarization(W, H=1, binary=False, stochastic=False):
    if not binary:
        Wb = W
    else:
        Wb = hard_sigmoid(W / H)
        if stochastic:
            # use hard sigmoid weight for possibility
            Wb = tf.contrib.keras.backend.random_binomial(tf.shape(Wb), p=Wb)
        else:
            # round weight to 0 and 1
            Wb = tf.round(Wb)
        # change range from 0~1  to  -1~1
        Wb = Wb*2-1
    return Wb


# shuffle data after one epoch
class data_generator(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.num_samples = x.shape[0]

    def data_gen(self, batch_size):
        x = self.x
        y = self.y
        num_batch = self.num_samples//batch_size
        batch_count = 0
        while 1:
            if batch_count < num_batch:
                a = batch_count*batch_size
                b = (batch_count+1)*batch_size
                batch_count += 1
                yield x[a:b, :], y[a:b]
            else:
                batch_count = 0
                mask = np.arange(self.num_samples)
                np.random.shuffle(mask)
                x = x[mask]
                y = y[mask]


class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, kernel_shape, padding="SAME", binarize_during_test=False, binary=False, stochastic=False,
                 is_training=None, index=0, fisher=False, fisherconst=.1, optimizer = 'Adam'):
        # binary: whether to implement the Binary Connect
        # stochastic: whether implement stochastic weight if do Binary Connect

        assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('conv_layer_%d' % index):
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                # the real value of weight
                weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer())

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape,
                                       initializer=tf.constant_initializer(0.))
                self.bias = bias

            # otherwise, return binarization directly
            # else:

            self.weight = weight
            self.wb = binarization(weight, H=1, binary=True, stochastic=stochastic)

            if binary:
                self.cell_out = tf.nn.conv2d(input_x, self.wb, strides=[1, 1, 1, 1], padding=padding)
            elif fisher:
                self.cell_out = tf.cond(is_training,  lambda: tf.nn.conv2d(input_x, self.weight, strides=[1, 1, 1, 1], padding=padding),
                                        lambda: tf.nn.conv2d(input_x, self.wb, strides=[1, 1, 1, 1], padding=padding))
            else:
                self.cell_out = tf.nn.conv2d(input_x, self.weight, strides=[1, 1, 1, 1], padding=padding)
            # if binary:
            #     cell_out = tf.nn.conv2d(input_x, self.wb, strides=[1, 1, 1, 1], padding=padding)
            # else:
            #     cell_out = tf.nn.conv2d(input_x, self.weight, strides=[1, 1, 1, 1], padding=padding)


            self.perturbation = self.weight - self.wb

            # if fisher:
            #     self.pertubation = tf.stop_gradient(weight - wb)
            self.fisher = fisher
            self.fisherconst = fisherconst
            self.binary = binary
            self.index=index

            self.cell_out = tf.add(self.cell_out, bias)


            tf.summary.histogram('conv_layer/{}/kernel'.format(index), self.weight)
            tf.summary.histogram('fc_layer/{}/binarized_kernel'.format(index), self.weight)
            tf.summary.histogram('conv_layer/{}/bias'.format(index), self.bias)
            tf.summary.histogram('conv_layer/{}/perturbation'.format(index), self.perturbation)

            if optimizer == 'Adam':
                # to store the moments for adam
                with tf.name_scope('conv_moment'):
                    self.m_w = tf.get_variable(name='conv_first_moment_w_%d' % index, shape=w_shape,
                                               initializer=tf.constant_initializer(0.))
                    self.v_w = tf.get_variable(name='conv_second_moment_w_%d' % index, shape=w_shape,
                                               initializer=tf.constant_initializer(0.))
                    self.m_b = tf.get_variable(name='conv_first_moment_b_%d' % index, shape=b_shape,
                                               initializer=tf.constant_initializer(0.))
                    self.v_b = tf.get_variable(name='conv_second_moment_b_%d' % index, shape=b_shape,
                                               initializer=tf.constant_initializer(0.))
    def output(self):
        return self.cell_out


class max_pooling_layer(object):
    def __init__(self, input_x, pool_size, padding="SAME"):

        k_size = pool_size
        with tf.variable_scope('max_pooling'):
            # strides [1, k_size, k_size, 1]
            pooling_shape = [1, k_size, k_size, 1]
            cell_out = tf.nn.max_pool(input_x, strides=pooling_shape, ksize=pooling_shape, padding=padding)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, binary=False, binarize_during_test = False, stochastic=False, is_training=None, index=0,
                 fisherconst=.1, fisher = False, optimizer = 'Adam'):
        # binary: whether to implement the Binary Connect
        # stochastic: whether implement stochastic weight if do Binary Connect

        with tf.variable_scope('fc_layer_%d' % index):
            with tf.name_scope('fc_kernel'):
                w_shape = [in_size, out_size]
                # the real value of weight
                weight = tf.get_variable(name='fc_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer())

            with tf.variable_scope('fc_kernel'):
                b_shape = [out_size]
                bias = tf.get_variable(name='fc_bias_%d' % index, shape=b_shape,
                                       initializer=tf.constant_initializer(0.))
                self.bias = bias

            self.wb = binarization(weight, H=1, binary=True, stochastic=stochastic)
            self.weight = weight
            #If we are regularizing the layer,
            # if binary or fisher:
            #     self.weight = tf.cond(is_training, lambda: tf.assign(weight, wb), lambda: tf.assign(weight,weight))
            # else:
            # otherwise, return binarization directly
            # else:

            # if binary:
            # else:
            # cell_out_b = tf.add(tf.matmul(input_x, self.wb), bias)
            # cell_out = tf.add(tf.matmul(input_x, self.weight), bias)



            if binary:
                self.cell_out = tf.add(tf.matmul(input_x, self.wb), bias)
            elif fisher:
                self.cell_out = tf.cond(is_training,  lambda: tf.add(tf.matmul(input_x, self.weight), bias),
                                        lambda: tf.add(tf.matmul(input_x, self.wb), bias))
                print(self.weight)
            else:
                self.cell_out = tf.add(tf.matmul(input_x, self.weight), bias)

            self.perturbation = self.weight - self.wb
            self.fisher = fisher
            self.fisherconst = fisherconst
            self.binary = binary
            self.index=index
            tf.summary.histogram('fc_layer/{}/kernel'.format(index), self.weight)
            tf.summary.histogram('fc_layer/{}/binarized_kernel'.format(index), self.weight)
            tf.summary.histogram('fc_layer/{}/bias'.format(index), self.bias)
            tf.summary.histogram('fc_layer/{}/perturbation'.format(index), self.perturbation)

            # to store the moments for adam
            if optimizer == 'Adam':
                with tf.name_scope('fc_moment'):
                    self.m_w = tf.get_variable(name='fc_first_moment_w_%d' % index, shape=w_shape,
                                               initializer=tf.constant_initializer(0.))
                    self.v_w = tf.get_variable(name='fc_second_moment_w_%d' % index, shape=w_shape,
                                               initializer=tf.constant_initializer(0.))
                    self.m_b = tf.get_variable(name='fc_first_moment_b_%d' % index, shape=b_shape,
                                               initializer=tf.constant_initializer(0.))
                    self.v_b = tf.get_variable(name='fc_second_moment_b_%d' % index, shape=b_shape,
                                               initializer=tf.constant_initializer(0.))

    def output(self):
        return self.cell_out


def l2_svm_loss(labels, net_output):
    loss = tf.reduce_mean(tf.square(tf.losses.hinge_loss(labels, net_output)))
    return loss


def tf_softmax_crossentropy_with_logits(labels, logits):
    print('Using Crossent Loss')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    return loss



class norm_layer(object):
    def __init__(self, input_x, is_training, is_drop_out, activation_function, index=0):
        with tf.variable_scope('batch_norm_%d' % index):
            cell_out = tf.contrib.layers.batch_norm(input_x, decay=0.99, updates_collections=None, is_training=is_training, epsilon=1e-4)

            # the activation function is after the batch norm
            if activation_function is not None:
                cell_out = activation_function(cell_out)

            # no dropout for CIFAR
            if is_drop_out:
                cell_out = tf.layers.dropout(cell_out, rate=0.0, training=is_training)

            self.cell_out = cell_out

    def output(self):
        return self.cell_out













