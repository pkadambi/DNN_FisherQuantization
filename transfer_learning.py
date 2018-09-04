import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import gradients_impl
import numpy as np
from train_data_ops import next_batch

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels
n_classes = 10
n_examples_per_class=1
inds = np.arange(len(x_train))
np.random.shuffle(inds)

# inds1[] =
# inds2[] =
# inds3[] =

y_train_class = y_train.argmax(axis=1)
train_indecies = []
small_train_set = []

for i in range(0, n_classes):
    train_indecies.append(np.array(np.where(y_train_class==i)))

i=0
train_indecies_ = []
for class_indecies in train_indecies:
    print('class: '+str(i))
    print(np.shape(class_indecies[0,0:n_examples_per_class]))
    small_train_set.append(class_indecies[0,0:n_examples_per_class])
    train_indecies_.append(np.array(class_indecies[0,n_examples_per_class:]))
    print(np.shape(class_indecies))
    i+=1

for class_indecies in train_indecies_:
    print(np.shape(class_indecies))
np.random.shuffle(train_indecies)
np.random.shuffle(train_indecies)
train_indecies = np.concatenate(train_indecies_)
print(np.shape(train_indecies))
ind = int(len(train_indecies)/2)
print(ind)

train_indecies1 = train_indecies[0:ind]
train_indecies2 = train_indecies[ind:]

print(np.shape(train_indecies1))
print(np.shape(train_indecies2))

x_train1 = x_train[train_indecies1,:]
x_train2 = x_train[train_indecies2,:]
print(np.shape(x_train1 ))
print(np.shape(x_train2 ))

y_train1 = y_train[train_indecies1,:]
y_train2 = y_train[train_indecies2,:]

small_train_set = np.array(small_train_set).flatten()
x_trainsmall = x_train[small_train_set ,:]
y_trainsmall =  y_train[small_train_set ,:]
print(np.shape(small_train_set))

print(train_indecies[0])

x_testcv = mnist.test.images
y_testcv = mnist.test.labels
# exit()


def two_layer_mlp_model(x_train, y_train, n_tot_iters=7500):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape = (None, 784), name='Inputs')
    y = tf.placeholder(tf.float32, shape = (None, 10), name='Labels')

    w = tf.get_variable(name='w', shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable(name='w2', shape = [512, 10], initializer = tf.contrib.layers.xavier_initializer())
    bias1 = tf.get_variable('bias1',shape= (512), initializer = tf.random_normal_initializer(stddev=.1))
    bias2 = tf.get_variable('bias2',shape= (10), initializer = tf.random_normal_initializer(stddev=.1))

    layer_1_out = tf.nn.relu(tf.matmul(x, w) + bias1)

    logits = tf.matmul(layer_1_out, w2) + bias2
    y_ = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.stop_gradient(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits))

    optimizer = tf.train.AdamOptimizer()
    ce_grads = tf.gradients(loss, [w, w2, bias1,bias2])
    train_op = optimizer.apply_gradients(zip(ce_grads, [w, w2, bias1, bias2]))
    vars = optimizer.variables()

    v_w2 = vars[-1]
    v_w1 = vars[-3]

    v_b2 = vars[-5]
    v_b1 = vars[-7]
    print(vars)

    n_iters = n_tot_iters
    batch_size = 1024
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    lossval=[]
    accval=[]
    sess.run(tf.global_variables_initializer())

    for i in range(0, n_iters):
        batch_data = mnist.train.next_batch(batch_size)

        x_batch = batch_data[0]
        y_batch = batch_data[1]

        _, l, acc, w_ = sess.run([train_op, loss, accuracy, w,], feed_dict={x: x_batch, y: y_batch})

        lossval.append(l)
        accval.append(acc)

        if i == n_iters-1:
            print('SAVING OPTIMAL WEIGHTS FROM END OF TRAINING')
            w1ml, w2ml = sess.run([w, w2])
            b1ml, b2ml = sess.run([bias1, bias2])
            fisher_weights1, fisher_weights2 = sess.run([v_w1, v_w2])
            fisher_biases1, fisher_biases2 = sess.run([v_b1, v_b2])

        if i%200==0:
            print('\nIteration: '+str(i)+'\nAccuracy: '+str(acc)+'\nLoss: '+str(l)+'\n')
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print('Test accuracy %g' % acc)

    weights1 = w1ml
    weights2 = w2ml


    biases1 = b1ml
    biases2 = b2ml


    sess.close()

    return fisher_weights1, fisher_weights2, weights1 , weights2, fisher_biases1, fisher_biases2, biases1, biases2, acc


def get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(fisher_weights1, fisher_weights2, fisher_biases1, fisher_biases2,
                                                                 weights1, weights2, biases1, biases2, x_train_, y_train_,
                                                                 reg_gamma=0., record_tensorboard=False, use_adam_for_regularizer=True):
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

    layer_1_out = tf.nn.relu(tf.matmul(x, w) + bias1)

    logits = tf.matmul(layer_1_out, w2) + bias2
    y_ = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.stop_gradient(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

    #model 1 pertubs
    w1_model1 = tf.placeholder(tf.float32, shape=(784,512))
    bias1_model1 = tf.placeholder(tf.float32, shape=(512))

    w2_model1 = tf.placeholder(tf.float32, shape=(512,10))
    bias2_model1 = tf.placeholder(tf.float32, shape=(10))

    # model 2 pertubs
    w1_model2 = tf.placeholder(tf.float32, shape=(784,512))
    bias1_model2 = tf.placeholder(tf.float32, shape=(512))

    w2_model2 = tf.placeholder(tf.float32, shape=(512,10))
    bias2_model2 = tf.placeholder(tf.float32, shape=(10))

    # w_pert = tf.stop_gradient(w + shift_pctage*w)



    optimizer = tf.train.AdamOptimizer()
    ce_grads = tf.gradients(loss, [w, w2, bias1,bias2])

    if reg_gamma>0.:
        # model1 perturbs--
        w1_pert1 = tf.stop_gradient(w - w1_model1)
        bias1_pert1 = tf.stop_gradient(bias1 - bias1_model1)

        w2_pert1 = tf.stop_gradient(w2 - w2_model1)
        bias2_pert1 = tf.stop_gradient(bias2 - bias2_model1)

        # model2 perturbs--
        w1_pert2 = tf.stop_gradient(w - w1_model2)
        bias1_pert2 = tf.stop_gradient(bias1 - bias1_model2)

        w2_pert2 = tf.stop_gradient(w2 - w2_model2)
        bias2_pert2 = tf.stop_gradient(bias2 - bias2_model2)
        #model1 regularizer gradient--
        reg_grad1_w1 = tf.multiply(fisher_weights1[0], w1_pert1)
        reg_grad1_biases1 = tf.multiply(fisher_biases1[0], bias1_pert1)

        reg_grad1_w2 = tf.multiply(fisher_weights2[0], w2_pert1)
        reg_grad1_biases2 = tf.multiply(fisher_biases2[0], bias2_pert1)

        #model2 regularizer gradient--
        reg_grad2_w1 = tf.multiply(fisher_weights1[1], w1_pert2)
        reg_grad2_biases1 = tf.multiply(fisher_biases1[1], bias1_pert2)

        reg_grad2_w2 = tf.multiply(fisher_weights2[1], w2_pert2)
        reg_grad2_biases2 = tf.multiply(fisher_biases2[1], bias2_pert2)


        if use_adam_for_regularizer:

            train_op = optimizer.apply_gradients(zip(ce_grads,[w, w2, bias1,bias2]))

            reg_grad_w1_tot = gamma * (reg_grad1_w1 + reg_grad2_w1)
            reg_grad_bias1_tot = gamma * (reg_grad2_biases1 + reg_grad1_biases1)

            reg_grad_w2_tot = gamma * (reg_grad1_w2 + reg_grad2_w2)
            reg_grad_bias2_tot = gamma *  (reg_grad1_biases2 + reg_grad2_biases2)

            train_op_reg = optimizer.apply_gradients(zip([reg_grad_w1_tot, reg_grad_w2_tot,
                                                      reg_grad_bias1_tot, reg_grad_bias2_tot],[w, w2, bias1,bias2]))


        else:

            reg_grad_w1_tot = gamma * (reg_grad1_w1 + reg_grad2_w1)
            reg_grad_bias1_tot = gamma * (reg_grad2_biases1 + reg_grad1_biases1)

            reg_grad_w2_tot = gamma * (reg_grad1_w2 + reg_grad2_w2)
            reg_grad_bias2_tot = gamma * (reg_grad1_biases2 + reg_grad2_biases2)

            tot_grad_w1 = ce_grads[0] + reg_grad_w1_tot
            tot_grad_bias1 = ce_grads[1] + reg_grad_w2_tot
            tot_grad_w2 = ce_grads[2] + reg_grad_bias1_tot
            tot_grad_bias2 = ce_grads[3] + reg_grad_bias2_tot

            train_op = optimizer.apply_gradients(zip([tot_grad_w1, tot_grad_bias1,
                                                      tot_grad_w2, tot_grad_bias2],[w, w2, bias1,bias2]))

            train_op_reg = tf.no_op()
    else:
        train_op = optimizer.apply_gradients(zip(ce_grads, [w, w2, bias1, bias2]))
        train_op_reg = tf.no_op()

    n_iters = 10
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    if record_tensorboard:
        summary_writer = tf.summary.FileWriter('./logs/two_layer_zero_mean', sess.graph)
        summary_op = tf.summary.merge_all()

    lossval=[]
    accval=[]
    sess.run(tf.global_variables_initializer())
    regularizer_const=reg_gamma

    for i in range(0,n_iters):
        _, __, l, acc = sess.run([train_op, train_op_reg, loss, accuracy],
                                 feed_dict={x: x_train_, y: y_train_, gamma:regularizer_const, w1_model1:weights1[0], bias1_model1:biases1[0],
                                            w2_model1:weights2[0], bias2_model1:biases2[0], w1_model2:weights1[1], bias1_model2:biases1[1],
                                            w2_model2:weights2[1], bias2_model2:biases2[1]})
        print('\nIteration: ' + str(i) + '\nAccuracy: ' + str(acc) + '\nLoss: ' + str(l) + '\n')



    if record_tensorboard:
        summ, _, __, l, acc = sess.run([summary_op, train_op, train_op_reg, loss, accuracy],
                                 feed_dict={x: x_train_, y: y_train_, gamma:regularizer_const, w1_model1:weights1[0], bias1_model1:biases1[0],
                                            w2_model1:weights2[0], bias2_model1:biases2[0], w1_model2:weights1[1], bias1_model2:biases1[1],
                                            w2_model2:weights2[1], bias2_model2:biases2[1]})
    if record_tensorboard:
        summary_writer.add_summary(summ, i)
    lossval.append(l)
    accval.append(acc)

    regularizer_const = 0.

    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print('Test accuracy %g' % acc)

    return acc

### Train Model 1
fisher_weights1, fisher_weights2, weights1 , weights2, fisher_biases1, fisher_biases2, biases1, biases2, acc = \
    two_layer_mlp_model(x_train1, y_train1)

### Train Model 2
fisher_weights1_, fisher_weights2_, weights1_ , weights2_, fisher_biases1_, fisher_biases2_, biases1_, biases2_, acc_  =\
    two_layer_mlp_model(x_train2, y_train2)
fisher_weights1=np.array([fisher_weights1, fisher_weights1_])
fisher_weights2=np.array([fisher_weights2, fisher_weights2_])
fisher_biases1=np.array([fisher_biases1, fisher_biases1_])
fisher_biases2=np.array([fisher_biases2, fisher_biases2_])
weights1=np.array([weights1, weights1_])
weights2=np.array([weights2, weights2_])
biases1=np.array([biases1, biases1_])
biases2=np.array([biases2, biases2_])
weights2, biases2, acc2 = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(fisher_weights1,
                                                               fisher_weights2, fisher_biases1,
                                                               fisher_biases2, weights1,
                                                               weights2 , biases1, biases2, x_trainsmall, y_trainsmall,reg_gamma=0.,
                                                                 record_tensorboard=False, use_adam_for_regularizer=True)


### Train Model





