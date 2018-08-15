from __future__ import print_function
import tensorflow as tf
import numpy as np
from cifar_utils import load_data
from matplotlib import pyplot as plt
from wt_ops import tf_hard_quantize, sigmoid_quantize, numpy_quantize, numpy_sigmoid_quantize, softmax
from train_data_ops import get_fashionMNIST, next_batch

dataset = get_fashionMNIST(convert_labels=False)

x_train = dataset[0]
y_train = dataset[1]

x_cv = dataset[2]
y_cv = dataset[3]

x_test = dataset[4]
y_test = dataset[5]

# BinaryConnect det.
from binary import fashiontraining
tf.reset_default_graph()
# binary = [0, 0, 0, 0, 0, 0, 0, 0, 0]
binary = [1, 1, 1,1]
fisher = [0, 0, 0, 0]
binarize_during_testing = [1, 1, 1, 1]
loss_type = 'l2svm'
fashiontraining(x_train, y_train, x_cv, y_cv, x_test, y_test,
         is_binary = binary,
         is_fisher = fisher,
         binarize_during_test = binarize_during_testing,
         is_stochastic = False,
         conv_featmap = [32, 64],
         fc_units = [7 * 7 * 64, 1024],
         conv_kernel_size = [5, 5],
         pooling_size = [2, 2],
         lr_start = 0.01,
         lr_end = 0.0003,
         epoch = 10 ,
         batch_size = 50,
         fisher_epochs=0,
         is_drop_out = False,
         verbose = True,
         pre_trained_model = None,
         loss_type=loss_type,
         record_tensorboard=True,
         logname='Fashionmnist_test')


