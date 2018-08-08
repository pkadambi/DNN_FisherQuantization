import numpy as np
import tensorflow as tf
import math
from tensorflow.python.framework import ops


def numpy_quantize(wts, qlevels=[-1.,1.], regions=None ):
    if regions == None:
        regions = [(qlevels[i] + qlevels[i + 1]) / 2. for i in range(len(qlevels) - 1)]
        # print('Quantizer levels were: '+ str(qlevels))
        # print('\n'+'Computed threshold regions: '+str(regions))

    n_regions = len(regions)
    wts_ = wts
    print(wts)
    for i in range(0, n_regions):
        if i ==0:
            inds = np.nonzero(wts_<=regions[i])
            # print(np.shape(inds))
            wts_[inds] = qlevels[i]
            # print(wts_)
            # print(wts_)
            # exit()
        else:
            inds = np.nonzero((~(wts<regions[i-1]))*(wts<=regions[i]))
            wts_[inds] = qlevels[i]
            # print(wts_)

        if i==n_regions-1:
            inds = np.nonzero(wts>regions[n_regions-1])
            # print('\nprinting inds' + str(inds))
            # print(inds)
            wts_[inds] = qlevels[-1]
            # print(wts_)
            return wts_
            # print(wts_)
            # exit()
    return wts_

def sigmoid(x):
  return 1 / (1 + math.exp(-x))



def softmax(z):
    EPSILON = np.finfo(float).eps
    z_ = z + EPSILON
    n_batch,n_classes = np.shape(z_)
    # print(np.shape(z_))
    z_ = np.exp(z_)

    summd = np.sum(z_,axis=1)
    for i in  range(0,n_batch):
        rw = z_[i,:]
        # print(rw / summd[i])
        z_[i,:] = rw/summd[i]
        # z_[i,:] = np.true_divide(np.exp(rw),denom)
    return z_

def sigmoid_quantize(w, levels, regions):

    output_levels = np.array(levels).reshape(np.shape(levels)[0], 1)
    input_regions = np.array(regions).reshape(np.shape(regions)[0], 1)

    alpha = 50

    qw = tf.stop_gradient(output_levels[0] + \
         (output_levels[1] - output_levels[0]) * tf.sigmoid(alpha * (w - input_regions[0]))) #+ \
         # (output_levels[2] - output_levels[1]) * tf.sigmoid(alpha * (w - input_regions[1])) )
         # (output_levels[3] - output_levels[2]) * tf.sigmoid(alpha * (w - input_regions[2])))
    return qw

def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        #x=tf.clip_by_value(x,-1,1)
        with g.gradient_override_map({"Sign": "Identity"}):
            return tf.sign(x)


def tf_hard_quantize(w, levels, regions):

    #TODO: IMPLEMENT FOR MORE THAN JUST BINARY
    if len(regions)==1:

        # for i in range(0, regions):
        ls_cmp = tf.cast(tf.stop_gradient(tf.less(w, regions[0])),tf.float32)
        # ls_cmp = tf.cast(tf.stop_gradient(tf.logical_not(tf.greater(w, regions[0]))),tf.float32)

        lower = tf.stop_gradient(ls_cmp  * tf.ones_like(w))
        w_q = tf.stop_gradient((levels[0]- levels[1])* tf.stop_gradient(tf.cast(lower,tf.float32)) + levels[1])
        return w_q
    return 0
    # elif len(regions)==2:


    # else:

def numpy_sigmoid_quantize():
    return 0
