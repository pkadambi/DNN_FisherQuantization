
import tensorflow as tf
import numpy as np
import time
from custom_optimizer import _adam_optimize_bn

from nn_framework import hard_sigmoid, binarization, data_generator, conv_layer, norm_layer, fc_layer, max_pooling_layer, l2_svm_loss, tf_softmax_crossentropy_with_logits


def Network(input_x, input_y, is_training, is_drop_out, is_binary, is_stochastic, binarize_during_test, channel_num, output_size,
            conv_featmap, fc_units, conv_kernel_size, pooling_size, learning_rate, is_fisher_regularized, gamma, loss_type, record_tensorboard):
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
                              binarize_during_test=binarize_during_test[0],
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
                              binarize_during_test=binarize_during_test[1],
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
                              binarize_during_test=binarize_during_test[2],
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
                              binarize_during_test=binarize_during_test[4],
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
                              binarize_during_test=binarize_during_test[5],
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
                              binarize_during_test=binarize_during_test[6],
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
                          binarize_during_test=binarize_during_test[7],
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
                          binarize_during_test=binarize_during_test[8],
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
    # with tf.name_scope("loss"):
    net_output = norm_layer_8.output()
    label = tf.one_hot(input_y, output_size)

    if loss_type == 'l2svm':
        # the hinge square loss
        loss = l2_svm_loss(labels=label, net_output=net_output)
    else:
        loss = tf_softmax_crossentropy_with_logits(labels = label, logits = net_output)

    tf.summary.histogram('Output', net_output)
    tf.summary.scalar('loss', loss)

    # net_output, loss, _updates = adam_optimize_bn()
    conv_layer_list = [conv_layer_0, conv_layer_1, conv_layer_2, conv_layer_3, conv_layer_4, conv_layer_5]
    fc_layer_list = [fc_layer_0, fc_layer_1, fc_layer_2]

    _updates = _adam_optimize_bn(loss, learning_rate, is_training, is_binary, conv_layer_list = conv_layer_list, fc_layer_list = fc_layer_list, gamma=gamma, record_tensorboard=record_tensorboard)

    # update parameters with adam

    return net_output, loss, _updates



# evaluate the output
def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=0)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('error_num', error_num)
    return error_num


def FashionNetwork(input_x, input_y, is_training, is_drop_out, is_binary, is_stochastic, binarize_during_test, channel_num, output_size,
            conv_featmap, fc_units, conv_kernel_size, pooling_size, learning_rate, is_fisher_regularized, gamma, loss_type='l2svm', record_tensorboard=False):
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
                              binarize_during_test = binarize_during_test[0],
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
                              binarize_during_test=binarize_during_test[1],
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
                          binary=is_binary[2],
                          binarize_during_test=binarize_during_test[2],
                          stochastic=is_stochastic,
                          is_training=is_training,
                          fisher=is_fisher_regularized[2],
                          index=0)
    norm_layer_8 = norm_layer(input_x=fc_layer_0.output(),
                              is_training=is_training,
                              is_drop_out=is_drop_out,
                              activation_function=tf.nn.relu,
                              index=8)
    fc_layer_1 = fc_layer(input_x=norm_layer_8.output(),
                          in_size=fc_units[1],
                          out_size=output_size,
                          binary=is_binary[2],
                          binarize_during_test=binarize_during_test[3],
                          stochastic=is_stochastic,
                          is_training=is_training,
                          fisher=is_fisher_regularized[3],
                          index=1)



    # compute loss
    # with tf.name_scope("loss"):
    net_output = fc_layer_1.output()
    # label = tf.one_hot(input_y, output_size)
    # the hinge square loss

    if loss_type == 'l2svm':
        # the hinge square loss
        loss = l2_svm_loss(labels=input_y, net_output=net_output)
    else:
        loss = tf_softmax_crossentropy_with_logits(labels=input_y, logits=net_output)


    tf.summary.histogram('net_output', net_output)
    tf.summary.scalar('loss', loss)

    conv_layer_list = [conv_layer_0, conv_layer_1]
    fc_layer_list = [fc_layer_0, fc_layer_1]

    _updates = _adam_optimize_bn(loss, learning_rate, is_training, is_binary, conv_layer_list=conv_layer_list,
                                 fc_layer_list=fc_layer_list, gamma=gamma, record_tensorboard=record_tensorboard)


    return net_output, loss, _updates


def training(X_train, y_train, X_val, y_val, X_test, y_test, is_binary, is_fisher, binarize_during_test, is_stochastic,
             conv_featmap, fc_units, conv_kernel_size, pooling_size, lr_start, lr_end, epoch, batch_size, is_drop_out,
             verbose=False, pre_trained_model=None, record_tensorboard=False, fisher_epochs = 0, loss_type='l2svm'):
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
    lr_decay = (lr_end / lr_start) ** (1. / epoch)
    lr_update = learning_rate.assign(tf.multiply(learning_rate, lr_decay))

    # build network
    output, loss, _update = Network(xs, ys, is_training,
                                     is_drop_out=is_drop_out,
                                     is_binary=is_binary,
                                     is_stochastic=is_stochastic,
                                     binarize_during_test=binarize_during_test,
                                     channel_num=3,
                                     output_size=10,
                                     conv_featmap=conv_featmap,
                                     fc_units=fc_units,
                                     conv_kernel_size=conv_kernel_size,
                                     pooling_size=pooling_size,
                                     learning_rate=learning_rate,
                                    is_fisher_regularized=is_fisher,
                                    gamma=gamma,
                                    loss_type=loss_type,
                                    record_tensorboard=record_tensorboard)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(ys, 1))
    accuracy = tf.stop_gradient(tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy'))

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
        iter_counter=0
        if pre_trained_model is None:
            fisher_gamma = 0.0
            for epc in range(epoch+fisher_epochs):
                print("epoch {} ".format(epc + 1))
                if epc>=(epoch):
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
                        summ, _, cur_loss, train_eve = sess.run([merge, _update, loss, correct_prediction ], feed_dict={xs: train_batch_x, ys: train_batch_y, is_training: True, gamma: fisher_gamma})
                        writer.add_summary(summ, iter_counter)
                    else:
                        _, cur_loss, train_eve = sess.run([_update, loss, correct_prediction ], feed_dict={xs: train_batch_x, ys: train_batch_y, is_training: True, gamma: fisher_gamma})

                    total_time += time.time()-start
                    train_eve_sum += np.sum(train_eve)
                    loss_sum += np.sum(cur_loss)
                    iter_counter+=1

                train_acc = 100 - train_eve_sum * 100 / y_train.shape[0]
                loss_sum /= iters
                print('average train loss: {} ,  average accuracy : {}%'.format(loss_sum, train_acc))

                valid_eve_sum = 0
                for i in range(y_val.shape[0]//val_batch_size):
                    val_batch_x = X_val[i*val_batch_size:(i+1)*val_batch_size]
                    val_batch_y = y_val[i*val_batch_size:(i+1)*val_batch_size]
                    valid_eve = sess.run([correct_prediction ], feed_dict={xs: val_batch_x, ys: val_batch_y, is_training: False})

                    valid_eve_sum += np.sum(valid_eve)

                valid_acc = 100 - valid_eve_sum * 100 / y_val.shape[0]

                _lr = sess.run([lr_update])
                print("updated learning rate: ", _lr)

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
                test_eve = sess.run([correct_prediction ], feed_dict={xs: test_batch_x, ys: test_batch_y, is_training: False})

                test_eve_sum += np.sum(test_eve)

            test_acc = 100 - test_eve_sum * 100 / y_test.shape[0]
            print('test accuracy: {}%'.format(test_acc))

            print("Traning ends. The best valid accuracy is {}%. Model named {}.".format(best_acc, cur_model_name))
        else:
            pass


def fashiontraining(X_train, y_train, X_val, y_val, X_test, y_test, is_binary, is_fisher, binarize_during_test, is_stochastic, conv_featmap,
             fc_units, conv_kernel_size, pooling_size, lr_start, lr_end, epoch, batch_size, is_drop_out, verbose=False,
             pre_trained_model=None, fisher_epochs=0, loss_type='l2svm', record_tensorboard=False, logname=None):
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
        ys = tf.placeholder(tf.float32, shape=(None, 10),  name='labels')
        gamma = tf.placeholder(shape=[], dtype=tf.float32)
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    # update learning rate
    num_training = y_train.shape[0]
    num_val = y_val.shape[0]
    learning_rate = tf.Variable(lr_start, name="learning_rate", dtype=tf.float32)
    lr_decay = (lr_end / lr_start) ** (1 / epoch)
    lr_update = learning_rate.assign(tf.multiply(learning_rate, lr_decay))

    # build network
    output, loss, _update = FashionNetwork(xs, ys, is_training,
                                    is_drop_out=is_drop_out,
                                    is_binary=is_binary,
                                    is_stochastic=is_stochastic,
                                    binarize_during_test=binarize_during_test,
                                    channel_num=1,
                                    output_size=10,
                                    conv_featmap=conv_featmap,
                                    fc_units=fc_units,
                                    conv_kernel_size=conv_kernel_size,
                                    pooling_size=pooling_size,
                                    learning_rate=learning_rate,
                                    is_fisher_regularized=is_fisher,
                                    gamma=gamma,
                                    loss_type=loss_type,
                                    record_tensorboard=record_tensorboard)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    # ys_index = tf.argmax(ys, axis=0)


    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(ys, 1))
    accuracy = tf.stop_gradient(tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy'))

    # batch size for validation, since validation set is too large
    val_batch_size = 100
    best_acc = 0

    cur_model_name = 'fashionmnist_{}'.format(int(time.time()))
    total_time = 0
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        if record_tensorboard:
            merge = tf.summary.merge_all()
            if logname==None:
                writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
            else:
                writer = tf.summary.FileWriter('log/'+logname, sess.graph)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer(), {is_training: False})

        iter_counter = 0
        # try to restore the pre_trained
        if pre_trained_model is None:
            fisher_gamma = 0.0
            for epc in range(epoch + fisher_epochs):
                print("epoch {} ".format(epc + 1))
                if epc>=(epoch):
                    print('Using fisher gamma')
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
                        summ, _, cur_loss, train_eve = sess.run([merge, _update, loss, correct_prediction],
                                                                feed_dict={xs: train_batch_x, ys: train_batch_y,
                                                                           is_training: True, gamma: fisher_gamma})
                        writer.add_summary(summ, iter_counter)
                    else:
                        _, cur_loss, train_eve = sess.run([_update, loss, correct_prediction],
                                                          feed_dict={xs: train_batch_x, ys: train_batch_y,
                                                                     is_training: True, gamma: fisher_gamma})

                    total_time += time.time() - start
                    train_eve_sum += np.sum(train_eve)
                    loss_sum += np.sum(cur_loss)
                    iter_counter += 1

                train_acc = train_eve_sum * 100 / y_train.shape[0]
                loss_sum /= iters
                print('average train loss: {} ,  average accuracy : {}%'.format(loss_sum, train_acc))

                valid_eve_sum = 0

                for i in range(y_val.shape[0] // val_batch_size):
                    val_batch_x = np.array(X_val[i * val_batch_size:(i + 1) * val_batch_size]).reshape(100,784)
                    val_batch_y = y_val[i * val_batch_size:(i + 1) * val_batch_size]

                    valid_eve = sess.run([correct_prediction], feed_dict={xs: val_batch_x, ys: val_batch_y, is_training: False})

                    valid_eve_sum += np.sum(valid_eve)

                valid_acc =  valid_eve_sum * 100 / y_val.shape[0]

                _lr = sess.run([lr_update])
                print("updated learning rate: ", _lr)

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
                test_eve = sess.run([correct_prediction], feed_dict={xs: test_batch_x, ys: test_batch_y, is_training: False})

                test_eve_sum += np.sum(test_eve)

            test_acc = test_eve_sum * 100 / y_test.shape[0]
            print('test accuracy: {}%'.format(test_acc))

            print("Traning ends. The best valid accuracy is {}%. Model named {}.".format(best_acc, cur_model_name))
        else:
            pass
