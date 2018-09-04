# a, b = get_acc_for_gaussian_perturbed_logistic_model_MNIST_diag(.0, .3, const_multiplier=.1, record_tensorboard=False)
# a, b = get_acc_for_gaussian_perturbed_logistic_model_MNIST(.0, .3, const_multiplier=.1, record_tensorboard=False)
# a, b = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(.02, .1, const_multiplier=0.005, record_tensorboard=False)
# a, b = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST_diag(.015, .03, const_multiplier=.005, record_tensorboard=False)
# a, b = get_acc_for_gaussian_perturbed_logistic_model_MNIST(.015, .005, const_multiplier=.33)


def get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(mu, sigma=.1, const_multiplier=1., n_tot_iters=5000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='hvp'):
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    from tensorflow.python.ops import gradients_impl
    import numpy as np
    tf.reset_default_graph()
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

    x = tf.placeholder(tf.float32, shape = (None, 784), name='Inputs')
    y = tf.placeholder(tf.float32, shape = (None, 10), name='Labels')
    gamma = tf.placeholder(tf.float32, shape = (), name='reg_constant')
    nwts = 7840
    # wts = tf.get_variable('Weights',shape= (784,10), initializer = tf.random_normal_initializer(stddev=.001))
    w = tf.get_variable(name='w', shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable(name='w2', shape = [512, 10], initializer = tf.contrib.layers.xavier_initializer())
    bias1 = tf.get_variable('bias1',shape= (512), initializer = tf.random_normal_initializer(stddev=.1))
    bias2 = tf.get_variable('bias2',shape= (10), initializer = tf.random_normal_initializer(stddev=.1))

    w_pert = tf.placeholder(tf.float32, shape=(784,512))
    w_pert2 = tf.placeholder(tf.float32, shape=(512,10))
    # 0.1000    0.1292    0.1668    0.2154    0.2783    0.3594    0.4642    0.5995    0.7743    1.0000

    # w_pert = tf.stop_gradient(w + shift_pctage*w)
    perturbation = tf.stop_gradient(w - w_pert)
    perturbation2 = tf.stop_gradient(w2 - w_pert2)


    layer_1_out = tf.nn.relu(tf.matmul(x, w) + bias1)

    logits = tf.matmul(layer_1_out, w2) + bias2
    y_ = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.stop_gradient(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits))


    optimizer = tf.train.AdamOptimizer()
    ce_grads = tf.gradients(loss, [w, w2, bias1,bias2])
    ce_grads_w1 = ce_grads[0]
    ce_grads_w2 = ce_grads[1]



    # print(vars)
    tf.summary.histogram('weights1', w)
    tf.summary.histogram('weights2', w2)
    tf.summary.histogram('pertweights1', w_pert)
    tf.summary.histogram('pertweights2', w_pert2)

    if regularizer_mode == 'hvp_adam':
        train_op = optimizer.apply_gradients(zip(ce_grads, [w, w2, bias1, bias2]))
        hvp1 = gradients_impl._hessian_vector_product(loss, [w], [perturbation])
        hvp2 = gradients_impl._hessian_vector_product(loss, [w2], [perturbation2])

        diag_load_amt1 = gamma * .005 * perturbation
        diag_load_amt2 = gamma * .005 * perturbation2

        reg_grad1 = gamma * 2.0 * hvp1 + diag_load_amt1
        reg_grad1 = tf.reshape(reg_grad1, tf.shape(w))
        reg_grad2 = gamma * 2.0 * hvp2 + diag_load_amt2
        reg_grad2 = tf.reshape(reg_grad2, tf.shape(w2))
        train_op_reg = optimizer.apply_gradients(zip([reg_grad1, reg_grad2], [w, w2]))

    elif regularizer_mode == 'diag_adam':
        train_op = optimizer.apply_gradients(zip(ce_grads, [w, w2, bias1, bias2]))
        vars = optimizer.variables()
        v_2 = vars[-1]
        v_1 = vars[-3]

        hvp1 = tf.multiply(v_1 ,perturbation)
        hvp2 = tf.multiply(v_2 ,perturbation2)

        diag_load_amt1 = gamma * .005 * perturbation
        diag_load_amt2 = gamma * .005 * perturbation2

        reg_grad1 = gamma * 2.0 * hvp1 + diag_load_amt1
        reg_grad1 = tf.reshape(reg_grad1, tf.shape(w))
        reg_grad2 = gamma * 2.0 * hvp2 + diag_load_amt2
        reg_grad2 = tf.reshape(reg_grad2, tf.shape(w2))
        train_op_reg = optimizer.apply_gradients(zip([reg_grad1, reg_grad2], [w, w2]))

    elif regularizer_mode == 'l2_adam':
        train_op = optimizer.apply_gradients(zip(ce_grads, [w, w2, bias1, bias2]))
        diag_load_amt1 = gamma * .005 * perturbation
        diag_load_amt2 = gamma * .005 * perturbation2

        reg_grad1 = diag_load_amt1
        reg_grad1 = tf.reshape(reg_grad1, tf.shape(w))
        reg_grad2 = diag_load_amt2
        reg_grad2 = tf.reshape(reg_grad2, tf.shape(w2))
        train_op_reg = optimizer.apply_gradients(zip([reg_grad1, reg_grad2], [w, w2]))

    elif regularizer_mode == 'hvp':
        diag_load_amt1 = gamma * .005 * perturbation
        diag_load_amt2 = gamma * .005 * perturbation2
        hvp1 = gradients_impl._hessian_vector_product(loss, [w], [perturbation])
        hvp2 = gradients_impl._hessian_vector_product(loss, [w2], [perturbation2])

        reg_grad1 = gamma * 2.0 * hvp1 + diag_load_amt1
        reg_grad1 = tf.reshape(reg_grad1, tf.shape(w))
        reg_grad2 = gamma * 2.0 * hvp2 + diag_load_amt2
        reg_grad2 = tf.reshape(reg_grad2, tf.shape(w2))
        tot_grads1 = ce_grads_w1 + reg_grad1
        tot_grads2 = ce_grads_w2 + reg_grad2
        train_op = optimizer.apply_gradients(zip([tot_grads1, tot_grads2, ce_grads[2], ce_grads[3]], [w, w2, bias1, bias2]))
        train_op_reg = tf.no_op()

    elif regularizer_mode == 'l2':
        diag_load_amt1 = gamma * .005 * perturbation
        diag_load_amt2 = gamma * .005 * perturbation2

        reg_grad1 = diag_load_amt1
        reg_grad1 = tf.reshape(reg_grad1, tf.shape(w))
        reg_grad2 = diag_load_amt2
        reg_grad2 = tf.reshape(reg_grad2, tf.shape(w2))
        tot_grads1 = ce_grads_w1 + reg_grad1
        tot_grads2 = ce_grads_w2 + reg_grad2
        train_op = optimizer.apply_gradients(zip([tot_grads1, tot_grads2, ce_grads[2], ce_grads[3]], [w, w2, bias1, bias2]))
        train_op_reg = tf.no_op()
    else:
        train_op = optimizer.apply_gradients(zip(ce_grads, [w, w2, bias1, bias2]))
        train_op_reg = tf.no_op()

    tf.summary.histogram('ce_gradient1', ce_grads_w1)
    tf.summary.histogram('ce_gradient2', ce_grads_w2)

    if const_multiplier>0.:
        print('USING REGULARIZATION')

        tf.summary.histogram('regularizer_gradient1', reg_grad1)
        tf.summary.histogram('regularizer_gradient2', reg_grad2)
        tf.summary.histogram('diagonal_load1', diag_load_amt1)
        tf.summary.histogram('diagonal_load2', diag_load_amt2)

        tf.summary.scalar('loss_gamma', gamma)
    else:
        print('NO REGULARIZATION')
        train_op_reg = tf.no_op()


    n_iters = n_tot_iters
    batch_size = 1024
    n_fisher_iters= n_fisher_iters
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    if record_tensorboard:
        summary_writer = tf.summary.FileWriter('./logs/two_layer_zero_mean', sess.graph)
        summary_op = tf.summary.merge_all()

    lossval=[]
    accval=[]
    sess.run(tf.global_variables_initializer())

    regularizer_const=0.
    w_pert_ = np.zeros([784, 512])
    w_pert2_ = np.zeros([512, 10])

    for i in range(0, n_iters):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        if i<=(n_iters-n_fisher_iters):
            regularizer_const=0.
        else:
            regularizer_const=.1*const_multiplier

        _, __, l, acc, w_ = sess.run([train_op, train_op_reg, loss, accuracy, w,], feed_dict={x: x_batch, y: y_batch, gamma:regularizer_const, w_pert:w_pert_, w_pert2:w_pert2_})

        if record_tensorboard:
            summ, _, __, l, acc, w_ = sess.run([summary_op, train_op, train_op_reg, loss, accuracy, w], feed_dict={x: x_batch, y: y_batch, gamma:regularizer_const, w_pert:w_pert_, w_pert2:w_pert2_})

        if record_tensorboard:
            summary_writer.add_summary(summ, i)
        lossval.append(l)
        accval.append(acc)

        if i == n_iters-n_fisher_iters:
            print('SAVING OPTIMAL ML WEIGHTS FROM END OF TRAINING')
            w_, w2_ = sess.run([w, w2])

        if i >= n_iters-n_fisher_iters and regularizer_const>0.:
            w_pert_ = w_ + np.random.normal(mu, sigma, size = [784, 512])
            w_pert2_ = w2_ + np.random.normal(mu, sigma, size = [512, 10])

        if i == n_iters - 1:
            print('USING PERTURBATIONS ON WEIGHTS AT END OF ALL ITERATIONS')
            w_, w2_ = sess.run([w, w2])
            # w_pert_ = w_
            # w_pert2_ = w2_
            # w_pert_ = w_ + np.random.normal(mu, sigma, size = [784, 512])

            # w_pert2_ = w2_ + np.random.normal(mu, sigma, size = [512, 10])


        if i%200==0:
            print('\nIteration: '+str(i)+'\nAccuracy: '+str(acc)+'\nLoss: '+str(l)+'\n')

    regularizer_const = 0.

    # perturbed_test_set = mnist.test.images+np.random.normal(0.,stddev, np.shape(mnist.test.images))
    w_pert_ = w_ + np.random.normal(mu, sigma, size = [784, 512])
    w_pert2_ = w2_ + np.random.normal(mu, sigma, size = [512, 10])

    x_testcv = mnist.test.images
    y_testcv = mnist.test.labels
    x_cv = x_testcv[0:5000,:]
    x_test = x_testcv[5000:,:]

    y_cv = y_testcv[0:5000,:]
    y_test = y_testcv[5000:,:]
    up_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print('UNPERTURBED Test accuracy %g' % up_acc)
    sess.run(tf.assign(w, w_pert), feed_dict={gamma:regularizer_const, w_pert:w_pert_, w_pert2: w_pert2_})
    sess.run(tf.assign(w2, w_pert2_), feed_dict={gamma:regularizer_const, w_pert:w_pert_, w_pert2: w_pert2_})

    pert_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, gamma:regularizer_const, w_pert:w_pert_, w_pert2:w_pert2_})
    # pert_acc = sess.run(accuracy, feed_dict={x: perturbed_test_set, y: mnist.test.labels})
    print('PRETURBED test accuracy %g' % pert_acc)
    # summary_writer.close()
    sess.close()

    return up_acc, pert_acc


# a, b = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(.02, .1, const_multiplier=0.00, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='hvp_adam')
# a, b = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(.05, .1, const_multiplier=0.05, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='hvp_adam')
# a, b = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(.005, .05, const_multiplier=0.00, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='none')
# a, b = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(.01, .005, const_multiplier=.05, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='none')
a, b = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(.1, .005, const_multiplier=.05, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='hvp_adam')
# a, b = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(.005, .0005, const_multiplier=0.0, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='hvp_adam')
# a, b = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(.005, .0005, const_multiplier=0.5, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='hvp_adam')
# a, b = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(.005, .05, const_multiplier=0.005, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='hvp_adam')
# a, b = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(.02, .05, const_multiplier=0.005, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='none')
# a, b = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(.2, .1, const_multiplier=0.0, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='none')



