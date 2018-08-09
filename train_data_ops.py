import tensorflow as tf
import numpy as np
from keras.datasets import fashion_mnist, cifar10, cifar100

def next_batch(data, labels, batch_size):
    '''
    Gets next batch for training
    Return a total of `batch_size` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    batch = [data_shuffle, labels_shuffle]
    return batch

def get_mnist(cross_validation_split = 0.5):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist_data = input_data.read_data_sets('/tmp/data', one_hot=True)
    return mnist_data

def get_cifar10(cross_validation_split = 0.5):
    '''
    Returns the cifar10 dataset

    :param: cross_validation_split
        Percentage of the test set to split off

    '''
    return 0

def get_cifar100(cross_validation_split = 0.5):
    return 0

def get_fashionMNIST(cross_validation_split = 0.5, convert_labels=True):
    '''
    Returns the fashionMNIST dataset

    :param: cross_validation_split
        Percentage of the test set to split off

    '''
    n_classes = 10
    (x_train, y_train), (x_test_validation, y_test_validation) = fashion_mnist.load_data()
    n_test, x_dim, y_dim = np.shape(x_test_validation)
    n_train, x_dim, y_dim = np.shape(x_train)

    x_train = x_train.astype('float32')/255
    x_test_validation = x_test_validation.astype('float32')/255

    x_train = x_train.reshape(n_train, x_dim*y_dim)
    x_test_validation = x_test_validation.reshape(n_test, x_dim*y_dim)
    indx = int(n_test / 2 - 1)

    if convert_labels:
        y_train = convert_label_to_one_hot(y_train, n_train, n_classes)
        y_test_validation = convert_label_to_one_hot(y_test_validation, n_test, n_classes)

        y_cv = y_test_validation[0:(n_test / 2 - 1), :]
        y_test = y_test_validation[(n_test/2):n_test,:]
        y_train = np.array(y_train).astype('float32')
        y_cv = np.array(y_cv).astype('float32')
        y_test = np.array(y_test).astype('float32')

    else:
        # print(np.shape(y_test_validation))
        y_train = np.array(y_train).astype('int64')
        y_test_validation = np.array(y_test_validation).astype('int64').reshape(n_test,1)
        y_test_validation = y_test_validation.reshape(n_test,1)
        print(np.shape(y_test_validation))
        y_cv = y_test_validation[0:indx ]
        y_test = y_test_validation[indx :n_test]


    x_cv = x_test_validation[indx,:]

    x_test = x_test_validation[indx, :]

    x_train = np.array(x_train).astype('float32')
    x_cv = np.array(x_cv).astype('float32')
    x_test = np.array(x_test).astype('float32')



    dataset = np.array([x_train, y_train, x_cv, y_cv, x_test, y_test])
    return dataset

def convert_label_to_one_hot(labels, n_samples, n_classes):
    '''
    takes an input label vector and returns an output that is one hot encoded

    ex:
    INPUT: [1, 0, 3]
    OUTPUT: array([[ 0.,  1.,  0.,  0.],
       [ 1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.]])
    '''

    one_hot_labels = np.zeros([n_samples, n_classes])
    one_hot_labels[np.arange(n_samples), labels] = 1

    return one_hot_labels




