# DNN_FisherQuantization
**Description:**
This is a framework for implementing a Fisher information based regularizer for ensuring that a neural network's weights are robust to quantization. Also, this framework contains an implementation of BinaryConnect to compare against as a baseline.

**Contains:**

    custom_optimizer.py - a custom implementation of ADAM that will work for this use case. Using TF's default ADAM seems to optimize over the unquantized weights instead of the quantized ones. ALSO: Calculates the fisher vector product.

    nn_framework.py - contains classes for the layers, binarization function that is used, and losses

    cifar_utils.py - loads/preprocesses the cifar dataset

    wt_ops.py - contains a variety of misc. functions
    
    networks.py - contains the networks themselves
    
    cifar10_binconnect.py - trains/tests the network described in the paper for CIFAR10
    
    fashionmnist.py - trains/tests a network for FashionMNIST
    
    mnist.py - trains/tests the MLP described in the paper on MNIST
    
    
    The other files contained in this repo are for old experiments, and serve only to revision control and mark them. 


Parts of this project are based on the implementation of binary connect given on github by one of the authors, MatthieuCourbariaux (he shared a Theano/Torch implementation), and also from user yy2779, who created a tensorflow version of MatthieuCourbariaux's code.

$\mathbf F$
