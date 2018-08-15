from __future__ import print_function
import tensorflow as tf
import numpy as np
from cifar_utils import load_data
from matplotlib import pyplot as plt

# Load the raw CIFAR-10 data.
X_train, y_train = load_data(mode='train')

mask = np.arange(X_train.shape[0])
np.random.shuffle(mask)
X_train = X_train[mask]
y_train = y_train[mask]

################## global contrast normalization ######################
lamda = 10
epsilon = 1e-7

#Because of memory issue, the pre-process of test data has to be after training data
mean_image = np.mean(X_train, axis=0)
X_train = X_train.astype(np.float32) - mean_image.astype(np.float32)
contrast = np.sqrt(lamda + np.mean(X_train**2,axis=0))
X_train = X_train / np.maximum(contrast, epsilon)

################## ZCA whitening #########################
temp = []
principal_components = []
for c in range(3):
    X = X_train[:,c*1024:(c+1)*1024]
    cov = np.dot(X.T, X) / (X.shape[0]-1)
    u, s, _ = np.linalg.svd(cov)
    principal_components.append( np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 10e-7))), u.T) )

    # Apply ZCA whitening
    whitex = np.dot(X, principal_components[c])
    temp.append(whitex)

X_train = np.append(temp[0],temp[1],axis=1)
X_train = np.append(X_train,temp[2],axis=1)
X_train = X_train.reshape([50000,3,32,32]).transpose((0,2,3,1))


num_training = 45000
num_validation = 5000

X_val = X_train[-num_validation:, :]
y_val = y_train[-num_validation:]

X_train = X_train[:num_training, :]
y_train = y_train[:num_training]

X_test = []
y_test = []

X_test, y_test = load_data(mode='test')
X_test = X_test.astype(np.float32) - mean_image
contrast = np.sqrt(lamda + np.mean(X_test**2,axis=0))
X_test = X_test / np.maximum(contrast, epsilon)

for c in range(3):
    X = X_test[:,c*1024:(c+1)*1024]
    whitex = np.dot(X, principal_components[c])
    temp[c] = whitex

X_test = np.append(temp[0],temp[1],axis=1)
X_test = np.append(X_test,temp[2],axis=1)
X_test = X_test.reshape([10000, 3, 32, 32]).transpose((0, 2, 3, 1))

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

del temp
del whitex
del principal_components
del u
del s
del X

# BinaryConnect det.
from binary import training
tf.reset_default_graph()
# binary = [0, 0, 0, 0, 0, 0, 0, 0, 0]
binary = [1, 1, 1, 1, 1, 1, 1, 1, 1]
fisher = [0, 0, 0, 0, 0, 0, 0, 0, 0]
binarize_during_testing = [1, 1, 1, 1, 1, 1, 1, 1, 1]
loss_type='l2svm'
training(X_train, y_train, X_val, y_val, X_test, y_test,
         is_binary = binary,
         is_fisher = fisher,
         binarize_during_test=binarize_during_testing,
         is_stochastic = False,
         conv_featmap = [128, 128, 256, 256, 512, 512],
         fc_units = [1024, 1024],
         conv_kernel_size = [3, 3, 3, 3, 3, 3],
         pooling_size = [2, 2, 2],
         lr_start = 0.003,
         lr_end = 0.000002,
         epoch = 50,
         batch_size = 50,
         fisher_epochs=5,
         is_drop_out = False,
         verbose = True,
         pre_trained_model = None,
         record_tensorboard=True,
         loss_type=loss_type)



