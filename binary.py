
import tensorflow as tf
import numpy as np
import time
from custom_optimizer import _adam_optimize_bn

from nn_framework import hard_sigmoid, binarization, data_generator, conv_layer, norm_layer, fc_layer, max_pooling_layer


def Network(input_x, input_y, is_training, is_drop_out, is_binary, is_stochastic, channel_num, output_size,
            conv_featmap, fc_units, conv_kernel_size, pooling_size, learning_rate, is_fisher_regularized, gamma):
    # is_training: whether train the network or validate it
    # is_drop_out: whether to dropout during training
    # is_binary: whether use the Binary Connect or not
    # is_stochastic: if use Binary Connect, whether to be stochastic
    # channel_num: input channel number, =3
    # output_size: 10 class
    # conv_featmap: number of features for convolution layer
    # fc_units: number of units for full connect layer
    # conv_kernel_size: kernel size for convolution layer
    # pooling_size: pooling size for max pooling layer
    # learning_rate: used for optimization

    # here is the architecture of the network
    # 128Conv3-BN-128Conv3-MaxPool2-BN-256Conv3-BN-256Conv3-MaxPool2-BN-512Conv3-BN-512Conv3-MaxPool2-BN-1024fc-1024fc-10fc
    
    conv_layer_0 = conv_layer(input_x=input_x,
                              in_channel=channel_num,
                              out_channel=conv_featmap[0],
                              kernel_shape=conv_kernel_size[0],
                              padding='VALID',
                              binary=is_binary[0],
                              stochastic=is_stochastic,
                              is_training=is_training,
                              fisher=is_fisher_regularized[0],
                              index=0)

    norm_layer_0 = norm_layer(input_x=conv_layer_0.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=tf.nn.relu,
                              index=0)

    conv_layer_1 = conv_layer(input_x=norm_layer_0.output(),
                              in_channel=conv_featmap[0],
                              out_channel=conv_featmap[1],
                              kernel_shape=conv_kernel_size[1],
                              padding='VALID',
                              binary=is_binary[1],
                              stochastic=is_stochastic,
                              is_training=is_training,
                              fisher=is_fisher_regularized[1],
                              index=1)

    pooling_layer_0 = max_pooling_layer(input_x=conv_layer_1.output(),
                                        pool_size=pooling_size[0],
                                        padding="SAME")

    norm_layer_1 = norm_layer(input_x=pooling_layer_0.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=tf.nn.relu,
                              index=1)

    conv_layer_2 = conv_layer(input_x=norm_layer_1.output(),
                              in_channel=conv_featmap[1],
                              out_channel=conv_featmap[2],
                              kernel_shape=conv_kernel_size[2],
                              padding='VALID',
                              binary=is_binary[2],
                              stochastic=is_stochastic,
                              is_training=is_training,
                              fisher=is_fisher_regularized[2],
                              index=2)

    norm_layer_2 = norm_layer(input_x=conv_layer_2.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=tf.nn.relu,
                              index=2)

    conv_layer_3 = conv_layer(input_x=norm_layer_2.output(),
                              in_channel=conv_featmap[2],
                              out_channel=conv_featmap[3],
                              kernel_shape=conv_kernel_size[3],
                              padding='VALID',
                              binary=is_binary[3],
                              stochastic=is_stochastic,
                              is_training=is_training,
                              fisher=is_fisher_regularized[3],
                              index=3)

    pooling_layer_1 = max_pooling_layer(input_x=conv_layer_3.output(),
                                        pool_size=pooling_size[1],
                                        padding="SAME")

    norm_layer_3 = norm_layer(input_x=pooling_layer_1.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=tf.nn.relu,
                              index=3)

    conv_layer_4 = conv_layer(input_x=norm_layer_3.output(),
                              in_channel=conv_featmap[3],
                              out_channel=conv_featmap[4],
                              kernel_shape=conv_kernel_size[4],
                              padding='VALID',
                              binary=is_binary[4],
                              stochastic=is_stochastic,
                              is_training=is_training,
                              fisher=is_fisher_regularized[4],
                              index=4)

    norm_layer_4 = norm_layer(input_x=conv_layer_4.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=tf.nn.relu,
                              index=4)

    conv_layer_5 = conv_layer(input_x=norm_layer_4.output(),
                              in_channel=conv_featmap[4],
                              out_channel=conv_featmap[5],
                              kernel_shape=conv_kernel_size[5],
                              padding='VALID',
                              binary=is_binary[5],
                              stochastic=is_stochastic,
                              is_training=is_training,
                              fisher=is_fisher_regularized[5],
                              index=5)

    pooling_layer_2 = max_pooling_layer(input_x=conv_layer_5.output(),
                                        pool_size=pooling_size[2],
                                        padding="SAME")

    norm_layer_5 = norm_layer(input_x=pooling_layer_2.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=tf.nn.relu,
                              index=5)

    # flatten the output of convolution layer
    pool_shape = norm_layer_5.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(norm_layer_5.output(), shape=[-1, img_vector_length])

    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=fc_units[0],
                          binary=is_binary[6],
                          stochastic=is_stochastic,
                          is_training=is_training,
                          fisher=is_fisher_regularized[6],
                          index=0)

    norm_layer_6 = norm_layer(input_x=fc_layer_0.output(),
                              is_training=is_training,
                              is_drop_out=is_drop_out,
                              activation_function=tf.nn.relu,
                              index=6)

    fc_layer_1 = fc_layer(input_x=norm_layer_6.output(),
                          in_size=fc_units[0],
                          out_size=fc_units[1],
                          binary=is_binary[7],
                          stochastic=is_stochastic,
                          is_training=is_training,
                          fisher=is_fisher_regularized[7],
                          index=1)

    norm_layer_7 = norm_layer(input_x=fc_layer_1.output(),
                              is_training=is_training,
                              is_drop_out=is_drop_out,
                              activation_function=tf.nn.relu,
                              index=7)

    fc_layer_2 = fc_layer(input_x=norm_layer_7.output(),
                          in_size=fc_units[1],
                          out_size=output_size,
                          binary=is_binary[8],
                          stochastic=is_stochastic,
                          is_training=is_training,
                          fisher=is_fisher_regularized[8],
                          index=2)

    norm_layer_8 = norm_layer(input_x=fc_layer_2.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=None,
                              index=8)

    # compute loss
    with tf.name_scope("loss"):
        net_output = norm_layer_8.output()
        label = tf.one_hot(input_y, output_size)
        # the hinge square loss
        loss = tf.reduce_mean(tf.square(tf.losses.hinge_loss(label, net_output)))
        tf.summary.scalar('loss', loss)

    # def adam_optimize_bn():
    #     with tf.name_scope("Adam_optimize"):
    #         beta1 = 0.9
    #         beta2 = 0.999
    #         epsilon = 1e-08
    #
    #         # time step
    #         t = tf.get_variable(name='timestep', shape=[], initializer=tf.constant_initializer(0))
    #
    #         # function that return all the updates
    #         def true_fn(loss=loss, conv_layer_0=conv_layer_0, conv_layer_1=conv_layer_1, conv_layer_2=conv_layer_2,
    #                     conv_layer_3=conv_layer_3, conv_layer_4=conv_layer_4, conv_layer_5=conv_layer_5,
    #                     fc_layer_0=fc_layer_0,
    #                     fc_layer_2=fc_layer_2, t=t):
    #
    #             new_t = t.assign(t + 1)
    #
    #             # calculate gradients
    #             grad_conv_wb0, grad_conv_wb1, grad_conv_wb2, grad_conv_wb3, grad_conv_wb4, grad_conv_wb5 \
    #                 = tf.gradients(ys=loss, xs=[conv_layer_0.wb, conv_layer_1.wb, conv_layer_2.wb, conv_layer_3.wb,
    #                                             conv_layer_4.wb, conv_layer_5.wb])
    #             grad_fc_wb0, grad_fc_wb1, grad_fc_wb2 \
    #                 = tf.gradients(ys=loss, xs=[fc_layer_0.wb, fc_layer_1.wb, fc_layer_2.wb])
    #
    #             grad_conv_b0, grad_conv_b1, grad_conv_b2, grad_conv_b3, grad_conv_b4, grad_conv_b5 \
    #                 = tf.gradients(ys=loss,
    #                                xs=[conv_layer_0.bias, conv_layer_1.bias, conv_layer_2.bias, conv_layer_3.bias,
    #                                    conv_layer_4.bias, conv_layer_5.bias])
    #             grad_fc_b0, grad_fc_b1, grad_fc_b2 \
    #                 = tf.gradients(ys=loss, xs=[fc_layer_0.bias, fc_layer_1.bias, fc_layer_2.bias])
    #
    #             # calculate updates for conv_layer_0
    #             new_conv_m_wb0 = conv_layer_0.m_w.assign(beta1 * conv_layer_0.m_w + (1. - beta1) * grad_conv_wb0)
    #             new_conv_v_wb0 = conv_layer_0.v_w.assign(beta2 * conv_layer_0.v_w + (1. - beta2) * grad_conv_wb0 ** 2)
    #             new_conv_m_b0 = conv_layer_0.m_b.assign(beta1 * conv_layer_0.m_b + (1. - beta1) * grad_conv_b0)
    #             new_conv_v_b0 = conv_layer_0.v_b.assign(beta2 * conv_layer_0.v_b + (1. - beta2) * grad_conv_b0 ** 2)
    #             update_conv_wb0 = new_conv_m_wb0 / (tf.sqrt(new_conv_v_wb0) + epsilon)
    #             update_conv_b0 = new_conv_m_b0 / (tf.sqrt(new_conv_v_b0) + epsilon)
    #
    #             # calculate updates for conv_layer_1
    #             new_conv_m_wb1 = conv_layer_1.m_w.assign(beta1 * conv_layer_1.m_w + (1. - beta1) * grad_conv_wb1)
    #             new_conv_v_wb1 = conv_layer_1.v_w.assign(beta2 * conv_layer_1.v_w + (1. - beta2) * grad_conv_wb1 ** 2)
    #             new_conv_m_b1 = conv_layer_1.m_b.assign(beta1 * conv_layer_1.m_b + (1. - beta1) * grad_conv_b1)
    #             new_conv_v_b1 = conv_layer_1.v_b.assign(beta2 * conv_layer_1.v_b + (1. - beta2) * grad_conv_b1 ** 2)
    #             update_conv_wb1 = new_conv_m_wb1 / (tf.sqrt(new_conv_v_wb1) + epsilon)
    #             update_conv_b1 = new_conv_m_b1 / (tf.sqrt(new_conv_v_b1) + epsilon)
    #
    #             # calculate updates for conv_layer_2
    #             new_conv_m_wb2 = conv_layer_2.m_w.assign(beta1 * conv_layer_2.m_w + (1. - beta1) * grad_conv_wb2)
    #             new_conv_v_wb2 = conv_layer_2.v_w.assign(beta2 * conv_layer_2.v_w + (1. - beta2) * grad_conv_wb2 ** 2)
    #             new_conv_m_b2 = conv_layer_2.m_b.assign(beta1 * conv_layer_2.m_b + (1. - beta1) * grad_conv_b2)
    #             new_conv_v_b2 = conv_layer_2.v_b.assign(beta2 * conv_layer_2.v_b + (1. - beta2) * grad_conv_b2 ** 2)
    #             update_conv_wb2 = new_conv_m_wb2 / (tf.sqrt(new_conv_v_wb2) + epsilon)
    #             update_conv_b2 = new_conv_m_b2 / (tf.sqrt(new_conv_v_b2) + epsilon)
    #
    #             # calculate updates for conv_layer_3
    #             new_conv_m_wb3 = conv_layer_3.m_w.assign(beta1 * conv_layer_3.m_w + (1. - beta1) * grad_conv_wb3)
    #             new_conv_v_wb3 = conv_layer_3.v_w.assign(beta2 * conv_layer_3.v_w + (1. - beta2) * grad_conv_wb3 ** 2)
    #             new_conv_m_b3 = conv_layer_3.m_b.assign(beta1 * conv_layer_3.m_b + (1. - beta1) * grad_conv_b3)
    #             new_conv_v_b3 = conv_layer_3.v_b.assign(beta2 * conv_layer_3.v_b + (1. - beta2) * grad_conv_b3 ** 2)
    #             update_conv_wb3 = new_conv_m_wb3 / (tf.sqrt(new_conv_v_wb3) + epsilon)
    #             update_conv_b3 = new_conv_m_b3 / (tf.sqrt(new_conv_v_b3) + epsilon)
    #
    #             # calculate updates for conv_layer_4
    #             new_conv_m_wb4 = conv_layer_4.m_w.assign(beta1 * conv_layer_4.m_w + (1. - beta1) * grad_conv_wb4)
    #             new_conv_v_wb4 = conv_layer_4.v_w.assign(beta2 * conv_layer_4.v_w + (1. - beta2) * grad_conv_wb4 ** 2)
    #             new_conv_m_b4 = conv_layer_4.m_b.assign(beta1 * conv_layer_4.m_b + (1. - beta1) * grad_conv_b4)
    #             new_conv_v_b4 = conv_layer_4.v_b.assign(beta2 * conv_layer_4.v_b + (1. - beta2) * grad_conv_b4 ** 2)
    #             update_conv_wb4 = new_conv_m_wb4 / (tf.sqrt(new_conv_v_wb4) + epsilon)
    #             update_conv_b4 = new_conv_m_b4 / (tf.sqrt(new_conv_v_b4) + epsilon)
    #
    #             # calculate updates for conv_layer_5
    #             new_conv_m_wb5 = conv_layer_5.m_w.assign(beta1 * conv_layer_5.m_w + (1. - beta1) * grad_conv_wb5)
    #             new_conv_v_wb5 = conv_layer_5.v_w.assign(beta2 * conv_layer_5.v_w + (1. - beta2) * grad_conv_wb5 ** 2)
    #             new_conv_m_b5 = conv_layer_5.m_b.assign(beta1 * conv_layer_5.m_b + (1. - beta1) * grad_conv_b5)
    #             new_conv_v_b5 = conv_layer_5.v_b.assign(beta2 * conv_layer_5.v_b + (1. - beta2) * grad_conv_b5 ** 2)
    #             update_conv_wb5 = new_conv_m_wb5 / (tf.sqrt(new_conv_v_wb5) + epsilon)
    #             update_conv_b5 = new_conv_m_b5 / (tf.sqrt(new_conv_v_b5) + epsilon)
    #
    #             # calculate updates for fc_layer_0
    #             new_fc_m_wb0 = fc_layer_0.m_w.assign(beta1 * fc_layer_0.m_w + (1. - beta1) * grad_fc_wb0)
    #             new_fc_v_wb0 = fc_layer_0.v_w.assign(beta2 * fc_layer_0.v_w + (1. - beta2) * grad_fc_wb0 ** 2)
    #             new_fc_m_b0 = fc_layer_0.m_b.assign(beta1 * fc_layer_0.m_b + (1. - beta1) * grad_fc_b0)
    #             new_fc_v_b0 = fc_layer_0.v_b.assign(beta2 * fc_layer_0.v_b + (1. - beta2) * grad_fc_b0 ** 2)
    #             update_fc_wb0 = new_fc_m_wb0 / (tf.sqrt(new_fc_v_wb0) + epsilon)
    #             update_fc_b0 = new_fc_m_b0 / (tf.sqrt(new_fc_v_b0) + epsilon)
    #
    #             # calculate updates for fc_layer_1
    #             new_fc_m_wb1 = fc_layer_1.m_w.assign(beta1 * fc_layer_1.m_w + (1. - beta1) * grad_fc_wb1)
    #             new_fc_v_wb1 = fc_layer_1.v_w.assign(beta2 * fc_layer_1.v_w + (1. - beta2) * grad_fc_wb1 ** 2)
    #             new_fc_m_b1 = fc_layer_1.m_b.assign(beta1 * fc_layer_1.m_b + (1. - beta1) * grad_fc_b1)
    #             new_fc_v_b1 = fc_layer_1.v_b.assign(beta2 * fc_layer_1.v_b + (1. - beta2) * grad_fc_b1 ** 2)
    #             update_fc_wb1 = new_fc_m_wb1 / (tf.sqrt(new_fc_v_wb1) + epsilon)
    #             update_fc_b1 = new_fc_m_b1 / (tf.sqrt(new_fc_v_b1) + epsilon)
    #
    #             # calculate updates for fc_layer_2
    #             new_fc_m_wb2 = fc_layer_2.m_w.assign(beta1 * fc_layer_2.m_w + (1. - beta1) * grad_fc_wb2)
    #             new_fc_v_wb2 = fc_layer_2.v_w.assign(beta2 * fc_layer_2.v_w + (1. - beta2) * grad_fc_wb2 ** 2)
    #             new_fc_m_b2 = fc_layer_2.m_b.assign(beta1 * fc_layer_2.m_b + (1. - beta1) * grad_fc_b2)
    #             new_fc_v_b2 = fc_layer_2.v_b.assign(beta2 * fc_layer_2.v_b + (1. - beta2) * grad_fc_b2 ** 2)
    #             update_fc_wb2 = new_fc_m_wb2 / (tf.sqrt(new_fc_v_wb2) + epsilon)
    #             update_fc_b2 = new_fc_m_b2 / (tf.sqrt(new_fc_v_b2) + epsilon)
    #
    #             return (update_conv_wb0, update_conv_wb1, update_conv_wb2, update_conv_wb3, update_conv_wb4,
    #                     update_conv_wb5,
    #                     update_fc_wb0, update_fc_wb1, update_fc_wb2, update_conv_b0, update_conv_b1, update_conv_b2,
    #                     update_conv_b3, update_conv_b4, update_conv_b5, update_fc_b0, update_fc_b1, update_fc_b2), new_t
    #
    #         # update = 0 in validation/test phase.
    #         def false_fn(t=t):
    #             return (0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.), t
    #
    #         # if is_training, do update
    #         adam_update, new_t = tf.cond(is_training, true_fn, false_fn)
    #
    #         # adjust learning rate with beta
    #         lr = learning_rate * tf.sqrt(1 - beta2 ** new_t) / (1 - beta1 ** new_t)
    #         tf.summary.scalar('learning_rate', lr)
    #
    #         # if is_binary, clip the weights to [-1, +1] before assignment
    #         if is_binary:
    #             new_conv_w0 = conv_layer_0.weight.assign(
    #                 tf.contrib.keras.backend.clip(conv_layer_0.weight - lr * adam_update[0], -1.0, 1.0))
    #             new_conv_w1 = conv_layer_1.weight.assign(
    #                 tf.contrib.keras.backend.clip(conv_layer_1.weight - lr * adam_update[1], -1.0, 1.0))
    #             new_conv_w2 = conv_layer_2.weight.assign(
    #                 tf.contrib.keras.backend.clip(conv_layer_2.weight - lr * adam_update[2], -1.0, 1.0))
    #             new_conv_w3 = conv_layer_3.weight.assign(
    #                 tf.contrib.keras.backend.clip(conv_layer_3.weight - lr * adam_update[3], -1.0, 1.0))
    #             new_conv_w4 = conv_layer_4.weight.assign(
    #                 tf.contrib.keras.backend.clip(conv_layer_4.weight - lr * adam_update[4], -1.0, 1.0))
    #             new_conv_w5 = conv_layer_5.weight.assign(
    #                 tf.contrib.keras.backend.clip(conv_layer_5.weight - lr * adam_update[5], -1.0, 1.0))
    #             new_fc_w0 = fc_layer_0.weight.assign(
    #                 tf.contrib.keras.backend.clip(fc_layer_0.weight - lr * adam_update[6], -1.0, 1.0))
    #             new_fc_w1 = fc_layer_1.weight.assign(
    #                 tf.contrib.keras.backend.clip(fc_layer_1.weight - lr * adam_update[7], -1.0, 1.0))
    #             new_fc_w2 = fc_layer_2.weight.assign(
    #                 tf.contrib.keras.backend.clip(fc_layer_2.weight - lr * adam_update[8], -1.0, 1.0))
    #         else:
    #             new_conv_w0 = conv_layer_0.weight.assign(conv_layer_0.weight - lr * adam_update[0])
    #             new_conv_w1 = conv_layer_1.weight.assign(conv_layer_1.weight - lr * adam_update[1])
    #             new_conv_w2 = conv_layer_2.weight.assign(conv_layer_2.weight - lr * adam_update[2])
    #             new_conv_w3 = conv_layer_3.weight.assign(conv_layer_3.weight - lr * adam_update[3])
    #             new_conv_w4 = conv_layer_4.weight.assign(conv_layer_4.weight - lr * adam_update[4])
    #             new_conv_w5 = conv_layer_5.weight.assign(conv_layer_5.weight - lr * adam_update[5])
    #             new_fc_w0 = fc_layer_0.weight.assign(fc_layer_0.weight - lr * adam_update[6])
    #             new_fc_w1 = fc_layer_1.weight.assign(fc_layer_1.weight - lr * adam_update[7])
    #             new_fc_w2 = fc_layer_2.weight.assign(fc_layer_2.weight - lr * adam_update[8])
    #
    #         new_conv_b0 = conv_layer_0.bias.assign(conv_layer_0.bias - lr * adam_update[9])
    #         new_conv_b1 = conv_layer_1.bias.assign(conv_layer_1.bias - lr * adam_update[10])
    #         new_conv_b2 = conv_layer_2.bias.assign(conv_layer_2.bias - lr * adam_update[11])
    #         new_conv_b3 = conv_layer_3.bias.assign(conv_layer_3.bias - lr * adam_update[12])
    #         new_conv_b4 = conv_layer_4.bias.assign(conv_layer_4.bias - lr * adam_update[13])
    #         new_conv_b5 = conv_layer_5.bias.assign(conv_layer_5.bias - lr * adam_update[14])
    #         new_fc_b0 = fc_layer_0.bias.assign(fc_layer_0.bias - lr * adam_update[15])
    #         new_fc_b1 = fc_layer_1.bias.assign(fc_layer_1.bias - lr * adam_update[16])
    #         new_fc_b2 = fc_layer_2.bias.assign(fc_layer_2.bias - lr * adam_update[17])
    #
    #     return net_output, loss, (new_conv_w0, new_conv_w1, new_conv_w2, new_conv_w3, new_conv_w4, new_conv_w5, new_fc_w0,
    #     new_fc_w1, new_fc_w2, new_conv_b0, new_conv_b1, new_conv_b2, new_conv_b3, new_conv_b4,
    #     new_conv_b5, new_fc_b0, new_fc_b1, new_fc_b2)

    # net_output, loss, _updates = adam_optimize_bn()
    conv_layer_list = [conv_layer_0, conv_layer_1, conv_layer_2, conv_layer_3, conv_layer_4, conv_layer_5]
    fc_layer_list = [fc_layer_0, fc_layer_1, fc_layer_2]

    _updates = _adam_optimize_bn(loss, learning_rate, is_training, is_binary, conv_layer_list = conv_layer_list, fc_layer_list = fc_layer_list, gamma=gamma)

    # update parameters with adam

    # with tf.name_scope("Adam_optimize"):
    #     beta1 = 0.9
    #     beta2 = 0.999
    #     epsilon = 1e-08
    #
    #     # time step
    #     t = tf.get_variable(name='timestep', shape=[], initializer=tf.constant_initializer(0))
    #
    #     # function that return all the updates
    #     def true_fn(loss=loss, conv_layer_0=conv_layer_0, conv_layer_1=conv_layer_1, conv_layer_2=conv_layer_2,
    #                 conv_layer_3=conv_layer_3, conv_layer_4=conv_layer_4, conv_layer_5=conv_layer_5, fc_layer_0=fc_layer_0,
    #                 fc_layer_2=fc_layer_2, t=t):
    #
    #         new_t = t.assign(t + 1)
    #
    #         # calculate gradients
    #         grad_conv_wb0, grad_conv_wb1, grad_conv_wb2, grad_conv_wb3, grad_conv_wb4, grad_conv_wb5 \
    #             = tf.gradients(ys=loss, xs=[conv_layer_0.wb, conv_layer_1.wb, conv_layer_2.wb, conv_layer_3.wb,
    #                                         conv_layer_4.wb, conv_layer_5.wb])
    #         grad_fc_wb0, grad_fc_wb1, grad_fc_wb2 \
    #             = tf.gradients(ys=loss, xs=[fc_layer_0.wb, fc_layer_1.wb, fc_layer_2.wb])
    #
    #         grad_conv_b0, grad_conv_b1, grad_conv_b2, grad_conv_b3, grad_conv_b4, grad_conv_b5 \
    #             = tf.gradients(ys=loss, xs=[conv_layer_0.bias, conv_layer_1.bias, conv_layer_2.bias, conv_layer_3.bias,
    #                                         conv_layer_4.bias, conv_layer_5.bias])
    #         grad_fc_b0, grad_fc_b1, grad_fc_b2 \
    #             = tf.gradients(ys=loss, xs=[fc_layer_0.bias, fc_layer_1.bias, fc_layer_2.bias])
    #
    #         # calculate updates for conv_layer_0
    #         new_conv_m_wb0 = conv_layer_0.m_w.assign(beta1 * conv_layer_0.m_w + (1. - beta1) * grad_conv_wb0)
    #         new_conv_v_wb0 = conv_layer_0.v_w.assign(beta2 * conv_layer_0.v_w + (1. - beta2) * grad_conv_wb0 ** 2)
    #         new_conv_m_b0 = conv_layer_0.m_b.assign(beta1 * conv_layer_0.m_b + (1. - beta1) * grad_conv_b0)
    #         new_conv_v_b0 = conv_layer_0.v_b.assign(beta2 * conv_layer_0.v_b + (1. - beta2) * grad_conv_b0 ** 2)
    #         update_conv_wb0 = new_conv_m_wb0 / (tf.sqrt(new_conv_v_wb0) + epsilon)
    #         update_conv_b0 = new_conv_m_b0 / (tf.sqrt(new_conv_v_b0) + epsilon)
    #
    #         # calculate updates for conv_layer_1
    #         new_conv_m_wb1 = conv_layer_1.m_w.assign(beta1 * conv_layer_1.m_w + (1. - beta1) * grad_conv_wb1)
    #         new_conv_v_wb1 = conv_layer_1.v_w.assign(beta2 * conv_layer_1.v_w + (1. - beta2) * grad_conv_wb1 ** 2)
    #         new_conv_m_b1 = conv_layer_1.m_b.assign(beta1 * conv_layer_1.m_b + (1. - beta1) * grad_conv_b1)
    #         new_conv_v_b1 = conv_layer_1.v_b.assign(beta2 * conv_layer_1.v_b + (1. - beta2) * grad_conv_b1 ** 2)
    #         update_conv_wb1 = new_conv_m_wb1 / (tf.sqrt(new_conv_v_wb1) + epsilon)
    #         update_conv_b1 = new_conv_m_b1 / (tf.sqrt(new_conv_v_b1) + epsilon)
    #
    #         # calculate updates for conv_layer_2
    #         new_conv_m_wb2 = conv_layer_2.m_w.assign(beta1 * conv_layer_2.m_w + (1. - beta1) * grad_conv_wb2)
    #         new_conv_v_wb2 = conv_layer_2.v_w.assign(beta2 * conv_layer_2.v_w + (1. - beta2) * grad_conv_wb2 ** 2)
    #         new_conv_m_b2 = conv_layer_2.m_b.assign(beta1 * conv_layer_2.m_b + (1. - beta1) * grad_conv_b2)
    #         new_conv_v_b2 = conv_layer_2.v_b.assign(beta2 * conv_layer_2.v_b + (1. - beta2) * grad_conv_b2 ** 2)
    #         update_conv_wb2 = new_conv_m_wb2 / (tf.sqrt(new_conv_v_wb2) + epsilon)
    #         update_conv_b2 = new_conv_m_b2 / (tf.sqrt(new_conv_v_b2) + epsilon)
    #
    #         # calculate updates for conv_layer_3
    #         new_conv_m_wb3 = conv_layer_3.m_w.assign(beta1 * conv_layer_3.m_w + (1. - beta1) * grad_conv_wb3)
    #         new_conv_v_wb3 = conv_layer_3.v_w.assign(beta2 * conv_layer_3.v_w + (1. - beta2) * grad_conv_wb3 ** 2)
    #         new_conv_m_b3 = conv_layer_3.m_b.assign(beta1 * conv_layer_3.m_b + (1. - beta1) * grad_conv_b3)
    #         new_conv_v_b3 = conv_layer_3.v_b.assign(beta2 * conv_layer_3.v_b + (1. - beta2) * grad_conv_b3 ** 2)
    #         update_conv_wb3 = new_conv_m_wb3 / (tf.sqrt(new_conv_v_wb3) + epsilon)
    #         update_conv_b3 = new_conv_m_b3 / (tf.sqrt(new_conv_v_b3) + epsilon)
    #
    #         # calculate updates for conv_layer_4
    #         new_conv_m_wb4 = conv_layer_4.m_w.assign(beta1 * conv_layer_4.m_w + (1. - beta1) * grad_conv_wb4)
    #         new_conv_v_wb4 = conv_layer_4.v_w.assign(beta2 * conv_layer_4.v_w + (1. - beta2) * grad_conv_wb4 ** 2)
    #         new_conv_m_b4 = conv_layer_4.m_b.assign(beta1 * conv_layer_4.m_b + (1. - beta1) * grad_conv_b4)
    #         new_conv_v_b4 = conv_layer_4.v_b.assign(beta2 * conv_layer_4.v_b + (1. - beta2) * grad_conv_b4 ** 2)
    #         update_conv_wb4 = new_conv_m_wb4 / (tf.sqrt(new_conv_v_wb4) + epsilon)
    #         update_conv_b4 = new_conv_m_b4 / (tf.sqrt(new_conv_v_b4) + epsilon)
    #
    #         # calculate updates for conv_layer_5
    #         new_conv_m_wb5 = conv_layer_5.m_w.assign(beta1 * conv_layer_5.m_w + (1. - beta1) * grad_conv_wb5)
    #         new_conv_v_wb5 = conv_layer_5.v_w.assign(beta2 * conv_layer_5.v_w + (1. - beta2) * grad_conv_wb5 ** 2)
    #         new_conv_m_b5 = conv_layer_5.m_b.assign(beta1 * conv_layer_5.m_b + (1. - beta1) * grad_conv_b5)
    #         new_conv_v_b5 = conv_layer_5.v_b.assign(beta2 * conv_layer_5.v_b + (1. - beta2) * grad_conv_b5 ** 2)
    #         update_conv_wb5 = new_conv_m_wb5 / (tf.sqrt(new_conv_v_wb5) + epsilon)
    #         update_conv_b5 = new_conv_m_b5 / (tf.sqrt(new_conv_v_b5) + epsilon)
    #
    #         # calculate updates for fc_layer_0
    #         new_fc_m_wb0 = fc_layer_0.m_w.assign(beta1 * fc_layer_0.m_w + (1. - beta1) * grad_fc_wb0)
    #         new_fc_v_wb0 = fc_layer_0.v_w.assign(beta2 * fc_layer_0.v_w + (1. - beta2) * grad_fc_wb0 ** 2)
    #         new_fc_m_b0 = fc_layer_0.m_b.assign(beta1 * fc_layer_0.m_b + (1. - beta1) * grad_fc_b0)
    #         new_fc_v_b0 = fc_layer_0.v_b.assign(beta2 * fc_layer_0.v_b + (1. - beta2) * grad_fc_b0 ** 2)
    #         update_fc_wb0 = new_fc_m_wb0 / (tf.sqrt(new_fc_v_wb0) + epsilon)
    #         update_fc_b0 = new_fc_m_b0 / (tf.sqrt(new_fc_v_b0) + epsilon)
    #
    #         # calculate updates for fc_layer_1
    #         new_fc_m_wb1 = fc_layer_1.m_w.assign(beta1 * fc_layer_1.m_w + (1. - beta1) * grad_fc_wb1)
    #         new_fc_v_wb1 = fc_layer_1.v_w.assign(beta2 * fc_layer_1.v_w + (1. - beta2) * grad_fc_wb1 ** 2)
    #         new_fc_m_b1 = fc_layer_1.m_b.assign(beta1 * fc_layer_1.m_b + (1. - beta1) * grad_fc_b1)
    #         new_fc_v_b1 = fc_layer_1.v_b.assign(beta2 * fc_layer_1.v_b + (1. - beta2) * grad_fc_b1 ** 2)
    #         update_fc_wb1 = new_fc_m_wb1 / (tf.sqrt(new_fc_v_wb1) + epsilon)
    #         update_fc_b1 = new_fc_m_b1 / (tf.sqrt(new_fc_v_b1) + epsilon)
    #
    #         # calculate updates for fc_layer_2
    #         new_fc_m_wb2 = fc_layer_2.m_w.assign(beta1 * fc_layer_2.m_w + (1. - beta1) * grad_fc_wb2)
    #         new_fc_v_wb2 = fc_layer_2.v_w.assign(beta2 * fc_layer_2.v_w + (1. - beta2) * grad_fc_wb2 ** 2)
    #         new_fc_m_b2 = fc_layer_2.m_b.assign(beta1 * fc_layer_2.m_b + (1. - beta1) * grad_fc_b2)
    #         new_fc_v_b2 = fc_layer_2.v_b.assign(beta2 * fc_layer_2.v_b + (1. - beta2) * grad_fc_b2 ** 2)
    #         update_fc_wb2 = new_fc_m_wb2 / (tf.sqrt(new_fc_v_wb2) + epsilon)
    #         update_fc_b2 = new_fc_m_b2 / (tf.sqrt(new_fc_v_b2) + epsilon)
    #
    #         return (update_conv_wb0, update_conv_wb1, update_conv_wb2, update_conv_wb3, update_conv_wb4, update_conv_wb5,
    #                 update_fc_wb0, update_fc_wb1, update_fc_wb2, update_conv_b0, update_conv_b1, update_conv_b2,
    #                 update_conv_b3, update_conv_b4, update_conv_b5, update_fc_b0, update_fc_b1, update_fc_b2), new_t
    #
    #     # update = 0 in validation/test phase.
    #     def false_fn(t=t):
    #         return (0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.), t
    #
    #     # if is_training, do update
    #     adam_update, new_t = tf.cond(is_training, true_fn, false_fn)
    #
    #     # adjust learning rate with beta
    #     lr = learning_rate * tf.sqrt(1 - beta2 ** new_t) / (1 - beta1 ** new_t)
    #     tf.summary.scalar('learning_rate',lr)
    #
    #     # if is_binary, clip the weights to [-1, +1] before assignment
    #     if is_binary:
    #         new_conv_w0 = conv_layer_0.weight.assign(tf.contrib.keras.backend.clip(conv_layer_0.weight - lr * adam_update[0], -1.0, 1.0))
    #         new_conv_w1 = conv_layer_1.weight.assign(tf.contrib.keras.backend.clip(conv_layer_1.weight - lr * adam_update[1], -1.0, 1.0))
    #         new_conv_w2 = conv_layer_2.weight.assign(tf.contrib.keras.backend.clip(conv_layer_2.weight - lr * adam_update[2], -1.0, 1.0))
    #         new_conv_w3 = conv_layer_3.weight.assign(tf.contrib.keras.backend.clip(conv_layer_3.weight - lr * adam_update[3], -1.0, 1.0))
    #         new_conv_w4 = conv_layer_4.weight.assign(tf.contrib.keras.backend.clip(conv_layer_4.weight - lr * adam_update[4], -1.0, 1.0))
    #         new_conv_w5 = conv_layer_5.weight.assign(tf.contrib.keras.backend.clip(conv_layer_5.weight - lr * adam_update[5], -1.0, 1.0))
    #         new_fc_w0 = fc_layer_0.weight.assign(tf.contrib.keras.backend.clip(fc_layer_0.weight - lr * adam_update[6], -1.0, 1.0))
    #         new_fc_w1 = fc_layer_1.weight.assign(tf.contrib.keras.backend.clip(fc_layer_1.weight - lr * adam_update[7], -1.0, 1.0))
    #         new_fc_w2 = fc_layer_2.weight.assign(tf.contrib.keras.backend.clip(fc_layer_2.weight - lr * adam_update[8], -1.0, 1.0))
    #     else:
    #         new_conv_w0 = conv_layer_0.weight.assign(conv_layer_0.weight - lr * adam_update[0])
    #         new_conv_w1 = conv_layer_1.weight.assign(conv_layer_1.weight - lr * adam_update[1])
    #         new_conv_w2 = conv_layer_2.weight.assign(conv_layer_2.weight - lr * adam_update[2])
    #         new_conv_w3 = conv_layer_3.weight.assign(conv_layer_3.weight - lr * adam_update[3])
    #         new_conv_w4 = conv_layer_4.weight.assign(conv_layer_4.weight - lr * adam_update[4])
    #         new_conv_w5 = conv_layer_5.weight.assign(conv_layer_5.weight - lr * adam_update[5])
    #         new_fc_w0 = fc_layer_0.weight.assign(fc_layer_0.weight - lr * adam_update[6])
    #         new_fc_w1 = fc_layer_1.weight.assign(fc_layer_1.weight - lr * adam_update[7])
    #         new_fc_w2 = fc_layer_2.weight.assign(fc_layer_2.weight - lr * adam_update[8])
    #
    #     new_conv_b0 = conv_layer_0.bias.assign(conv_layer_0.bias - lr * adam_update[9])
    #     new_conv_b1 = conv_layer_1.bias.assign(conv_layer_1.bias - lr * adam_update[10])
    #     new_conv_b2 = conv_layer_2.bias.assign(conv_layer_2.bias - lr * adam_update[11])
    #     new_conv_b3 = conv_layer_3.bias.assign(conv_layer_3.bias - lr * adam_update[12])
    #     new_conv_b4 = conv_layer_4.bias.assign(conv_layer_4.bias - lr * adam_update[13])
    #     new_conv_b5 = conv_layer_5.bias.assign(conv_layer_5.bias - lr * adam_update[14])
    #     new_fc_b0 = fc_layer_0.bias.assign(fc_layer_0.bias - lr * adam_update[15])
    #     new_fc_b1 = fc_layer_1.bias.assign(fc_layer_1.bias - lr * adam_update[16])
    #     new_fc_b2 = fc_layer_2.bias.assign(fc_layer_2.bias - lr * adam_update[17])
    #
    return net_output, loss, _updates



# evaluate the output
def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('error_num', error_num)
    return error_num


def FashionNetwork(input_x, input_y, is_training, is_drop_out, is_binary, is_stochastic, channel_num, output_size,
            conv_featmap, fc_units, conv_kernel_size, pooling_size, learning_rate, is_fisher_regularized, gamma):
    # is_training: whether train the network or validate it
    # is_drop_out: whether to dropout during training
    # is_binary: whether use the Binary Connect or not
    # is_stochastic: if use Binary Connect, whether to be stochastic
    # channel_num: input channel number, =3
    # output_size: 10 class
    # conv_featmap: number of features for convolution layer
    # fc_units: number of units for full connect layer
    # conv_kernel_size: kernel size for convolution layer
    # pooling_size: pooling size for max pooling layer
    # learning_rate: used for optimization

    # here is the architecture of the network
    # 128Conv3-BN-128Conv3-MaxPool2-BN-256Conv3-BN-256Conv3-MaxPool2-BN-512Conv3-BN-512Conv3-MaxPool2-BN-1024fc-1024fc-10fc
    x_image = tf.reshape(input_x, [-1, 28, 28, 1])

    conv_layer_0 = conv_layer(input_x=x_image,
                              in_channel=channel_num,
                              out_channel=conv_featmap[0],
                              kernel_shape=conv_kernel_size[0],
                              padding='SAME',
                              binary=is_binary[0],
                              stochastic=is_stochastic,
                              is_training=is_training,
                              fisher=is_fisher_regularized[0],
                              index=0)
    pooling_layer_0 = max_pooling_layer(input_x=conv_layer_0.output(),
                                        pool_size=pooling_size[0],
                                        padding="SAME")


    conv_layer_1 = conv_layer(input_x=pooling_layer_0.output(),
                              in_channel=conv_featmap[0],
                              out_channel=conv_featmap[1],
                              kernel_shape=conv_kernel_size[1],
                              padding='SAME',
                              binary=is_binary[1],
                              stochastic=is_stochastic,
                              is_training=is_training,
                              fisher=is_fisher_regularized[1],
                              index=1)

    pooling_layer_1 = max_pooling_layer(input_x=conv_layer_1.output(),
                                        pool_size=pooling_size[1],
                                        padding="SAME")


    # flatten the output of convolution layer
    # pool_shape = pooling_layer_1.output().get_shape()
    # img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    # print(fc_units[0])
    flatten = tf.reshape(pooling_layer_1.output(), [-1,fc_units[0]])
    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=fc_units[0],
                          out_size=fc_units[1],
                          binary=is_binary[6],
                          stochastic=is_stochastic,
                          is_training=is_training,
                          fisher=is_fisher_regularized[6],
                          index=0)
    norm_layer_8 = norm_layer(input_x=fc_layer_0.output(),
                              is_training=is_training,
                              is_drop_out=is_drop_out,
                              activation_function=tf.nn.relu,
                              index=8)
    fc_layer_1 = fc_layer(input_x=norm_layer_8.output(),
                          in_size=fc_units[1],
                          out_size=output_size,
                          binary=is_binary[7],
                          stochastic=is_stochastic,
                          is_training=is_training,
                          fisher=is_fisher_regularized[7],
                          index=1)



    # compute loss
    with tf.name_scope("loss"):
        net_output =     fc_layer_1.output()
        label = tf.one_hot(input_y, output_size)
        # the hinge square loss
        loss = tf.reduce_mean(tf.square(tf.losses.hinge_loss(label, net_output)))
        tf.summary.scalar('loss', loss)

    conv_layer_list = [conv_layer_0, conv_layer_1]
    fc_layer_list = [fc_layer_0, fc_layer_1]

    _updates = _adam_optimize_bn(loss, learning_rate, is_training, is_binary, conv_layer_list=conv_layer_list,
                                 fc_layer_list=fc_layer_list, gamma=gamma)


    return net_output, loss, _updates


def training(X_train, y_train, X_val, y_val, X_test, y_test, is_binary, is_fisher, is_stochastic, conv_featmap, fc_units, conv_kernel_size, pooling_size, lr_start, lr_end, epoch, batch_size, is_drop_out, verbose=False, pre_trained_model=None, record_tensorboard=False, fisher_epochs = 0):
    # X_train, y_train, X_val, y_val, X_test, y_test:
    # is_binary: whether use the Binary Connect or not
    # is_stochastic: if use Binary Connect, whether to be stochastic
    # conv_featmap: number of features for convolution layer
    # fc_units: number of units for full connect layer
    # conv_kernel_size: kernel size for convolution layer
    # pooling_size: pooling size for max pooling layer
    # lr_start: init learning rate
    # lr_end: final learning rate, used for calculate lr_decay
    # epoch: times of training
    # batch_size; training batch size
    # is_drop_out: whether to dropout during training

    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
        gamma = tf.placeholder(shape=[], dtype=tf.float32)
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    
    # update learning rate
    num_training = y_train.shape[0]
    num_val = y_val.shape[0]
    learning_rate = tf.Variable(lr_start, name="learning_rate", dtype=tf.float32)
    # lr_decay = (lr_end / lr_start) ** (1 / epoch)
    # lr_update = learning_rate.assign(tf.multiply(learning_rate, lr_decay))

    # build network
    output, loss, _update = Network(xs, ys, is_training,
                                     is_drop_out=is_drop_out,
                                     is_binary=is_binary,
                                     is_stochastic=is_stochastic,
                                     channel_num=3,
                                     output_size=10,
                                     conv_featmap=conv_featmap,
                                     fc_units=fc_units,
                                     conv_kernel_size=conv_kernel_size,
                                     pooling_size=pooling_size,
                                     learning_rate=learning_rate,
                                    is_fisher_regularized=is_fisher,
                                    gamma=gamma)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    eve = evaluate(output, ys)
    
    # batch size for validation, since validation set is too large
    val_batch_size = 100 
    best_acc = 0
    cur_model_name = 'cifar10_{}'.format(int(time.time()))
    total_time = 0

    with tf.Session() as sess:

        if record_tensorboard:

            merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer(), {is_training: False})

        # try to restore the pre_trained
        if pre_trained_model is None:
            i=0
            fisher_gamma = 0.0
            for epc in range(epoch+fisher_epochs):
                print("epoch {} ".format(epc + 1))
                if epc>=(epoch+fisher_epochs):
                    fisher_gamma=1.
                train_eve_sum = 0
                loss_sum = 0
                for _ in range(iters + 1):
                    # randomly choose train data
                    mask = np.random.choice(num_training, batch_size, replace=False)
                    np.random.shuffle(mask)
                    train_batch_x = X_train[mask]
                    train_batch_y = y_train[mask]

                    start = time.time()

                    if record_tensorboard:
                        summ, _, cur_loss, train_eve = sess.run([merge, _update, loss, eve], feed_dict={xs: train_batch_x, ys: train_batch_y, is_training: True, gamma: fisher_gamma})
                        writer.add_summary(summ, i)
                    else:
                        _, cur_loss, train_eve = sess.run([_update, loss, eve], feed_dict={xs: train_batch_x, ys: train_batch_y, is_training: True, gamma: fisher_gamma})

                    total_time += time.time()-start
                    train_eve_sum += np.sum(train_eve)
                    loss_sum += np.sum(cur_loss)
                    i+=1

                train_acc = 100 - train_eve_sum * 100 / y_train.shape[0]
                loss_sum /= iters
                print('average train loss: {} ,  average accuracy : {}%'.format(loss_sum, train_acc))

                valid_eve_sum = 0
                for i in range(y_val.shape[0]//val_batch_size):
                    val_batch_x = X_val[i*val_batch_size:(i+1)*val_batch_size]
                    val_batch_y = y_val[i*val_batch_size:(i+1)*val_batch_size]
                    valid_eve = sess.run([eve], feed_dict={xs: val_batch_x, ys: val_batch_y, is_training: False})

                    valid_eve_sum += np.sum(valid_eve)

                valid_acc = 100 - valid_eve_sum * 100 / y_val.shape[0]

                # _lr = sess.run([lr_update])
                # print("updated learning rate: ", _lr)

                if verbose:
                    print('validation accuracy : {}%'.format(valid_acc))

                # save the merge result summary

                # when achieve the best validation accuracy, we store the model paramters
                if valid_acc > best_acc:
                    print('* Best accuracy: {}%'.format(valid_acc))
                    best_acc = valid_acc
                    saver.save(sess, 'model/{}'.format(cur_model_name))

            # test the network
            test_eve_sum = 0
            for i in range(y_test.shape[0] // 100 + 1):
                a = i * 100
                if a >= y_test.shape[0]:
                    continue
                b = (i + 1) * 100 if (i + 1) * 100 < y_test.shape[0] else y_test.shape[0]
                test_batch_x = X_test[a:b]
                test_batch_y = y_test[a:b]
                test_eve = sess.run([eve], feed_dict={xs: test_batch_x, ys: test_batch_y, is_training: False})

                test_eve_sum += np.sum(test_eve)

            test_acc = 100 - test_eve_sum * 100 / y_test.shape[0]
            print('test accuracy: {}%'.format(test_acc))

            print("Traning ends. The best valid accuracy is {}%. Model named {}.".format(best_acc, cur_model_name))
        else:
            pass


def fashiontraining(X_train, y_train, X_val, y_val, X_test, y_test, is_binary, is_fisher, is_stochastic, conv_featmap,
             fc_units, conv_kernel_size, pooling_size, lr_start, lr_end, epoch, batch_size, is_drop_out, verbose=False,
             pre_trained_model=None, record_tensorboard=False, fisher_epochs=0):
    # X_train, y_train, X_val, y_val, X_test, y_test:
    # is_binary: whether use the Binary Connect or not
    # is_stochastic: if use Binary Connect, whether to be stochastic
    # conv_featmap: number of features for convolution layer
    # fc_units: number of units for full connect layer
    # conv_kernel_size: kernel size for convolution layer
    # pooling_size: pooling size for max pooling layer
    # lr_start: init learning rate
    # lr_end: final learning rate, used for calculate lr_decay
    # epoch: times of training
    # batch_size; training batch size
    # is_drop_out: whether to dropout during training

    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        # x = tf.placeholder(tf.float32, shape=(None, 784), name='inputs')
        # y_ = tf.placeholder(tf.float32, shape=(None ), name='labels')
        xs = tf.placeholder(shape=[None, 784], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
        gamma = tf.placeholder(shape=[], dtype=tf.float32)
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    # update learning rate
    num_training = y_train.shape[0]
    num_val = y_val.shape[0]
    learning_rate = tf.Variable(lr_start, name="learning_rate", dtype=tf.float32)
    # lr_decay = (lr_end / lr_start) ** (1 / epoch)
    # lr_update = learning_rate.assign(tf.multiply(learning_rate, lr_decay))

    # build network
    output, loss, _update = FashionNetwork(xs, ys, is_training,
                                    is_drop_out=is_drop_out,
                                    is_binary=is_binary,
                                    is_stochastic=is_stochastic,
                                    channel_num=1,
                                    output_size=10,
                                    conv_featmap=conv_featmap,
                                    fc_units=fc_units,
                                    conv_kernel_size=conv_kernel_size,
                                    pooling_size=pooling_size,
                                    learning_rate=learning_rate,
                                    is_fisher_regularized=is_fisher,
                                    gamma=gamma)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    eve = evaluate(output, ys)

    # batch size for validation, since validation set is too large
    val_batch_size = 100
    best_acc = 0
    cur_model_name = 'cifar10_{}'.format(int(time.time()))
    total_time = 0

    with tf.Session() as sess:

        if record_tensorboard:
            merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer(), {is_training: False})

        # try to restore the pre_trained
        if pre_trained_model is None:
            i = 0
            fisher_gamma = 0.0
            for epc in range(epoch + fisher_epochs):
                print("epoch {} ".format(epc + 1))
                if epc >= (epoch + fisher_epochs):
                    fisher_gamma = 1.
                train_eve_sum = 0
                loss_sum = 0
                for _ in range(iters + 1):
                    # randomly choose train data
                    mask = np.random.choice(num_training, batch_size, replace=False)
                    np.random.shuffle(mask)
                    train_batch_x = X_train[mask]
                    train_batch_y = y_train[mask]

                    start = time.time()

                    if record_tensorboard:
                        summ, _, cur_loss, train_eve = sess.run([merge, _update, loss, eve],
                                                                feed_dict={xs: train_batch_x, ys: train_batch_y,
                                                                           is_training: True, gamma: fisher_gamma})
                        writer.add_summary(summ, i)
                    else:
                        _, cur_loss, train_eve = sess.run([_update, loss, eve],
                                                          feed_dict={xs: train_batch_x, ys: train_batch_y,
                                                                     is_training: True, gamma: fisher_gamma})

                    total_time += time.time() - start
                    train_eve_sum += np.sum(train_eve)
                    loss_sum += np.sum(cur_loss)
                    i += 1

                train_acc = 100 - train_eve_sum * 100 / y_train.shape[0]
                loss_sum /= iters
                print('average train loss: {} ,  average accuracy : {}%'.format(loss_sum, train_acc))

                valid_eve_sum = 0
                for i in range(y_val.shape[0] // val_batch_size):
                    val_batch_x = np.array(X_val[i * val_batch_size:(i + 1) * val_batch_size]).reshape(100,784)
                    val_batch_y = y_val[i * val_batch_size:(i + 1) * val_batch_size]

                    valid_eve = sess.run([eve], feed_dict={xs: val_batch_x, ys: val_batch_y, is_training: False})

                    valid_eve_sum += np.sum(valid_eve)

                valid_acc = 100 - valid_eve_sum * 100 / y_val.shape[0]

                # _lr = sess.run([lr_update])
                # print("updated learning rate: ", _lr)

                if verbose:
                    print('validation accuracy : {}%'.format(valid_acc))

                # save the merge result summary

                # when achieve the best validation accuracy, we store the model paramters
                if valid_acc > best_acc:
                    print('* Best accuracy: {}%'.format(valid_acc))
                    best_acc = valid_acc
                    saver.save(sess, 'model/{}'.format(cur_model_name))

            # test the network
            test_eve_sum = 0
            for i in range(y_test.shape[0] // 100 + 1):
                a = i * 100
                if a >= y_test.shape[0]:
                    continue
                b = (i + 1) * 100 if (i + 1) * 100 < y_test.shape[0] else y_test.shape[0]
                test_batch_x = X_test[a:b]
                test_batch_y = y_test[a:b]
                test_eve = sess.run([eve], feed_dict={xs: test_batch_x, ys: test_batch_y, is_training: False})

                test_eve_sum += np.sum(test_eve)

            test_acc = 100 - test_eve_sum * 100 / y_test.shape[0]
            print('test accuracy: {}%'.format(test_acc))

            print("Traning ends. The best valid accuracy is {}%. Model named {}.".format(best_acc, cur_model_name))
        else:
            pass
