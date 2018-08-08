'''
Provide custom implementation of Adam.

Motivation:
the following pseudocode doesnt work
    optimizer.update(w_full_precision, tf.gradients_via_adam(loss, w_b))

    what the above line does is create the momentum vars w.r.t the full precision wts
    but, we want the momentum vars wrt the binarized wts
'''

import tensorflow as tf
from tensorflow.python.ops import gradients_impl

def _adam_optimize_bn(loss, learning_rate, is_training, is_binary, gamma=0.0, conv_layer_list=None, fc_layer_list=None, ):
    with tf.name_scope("Adam_optimize"):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-08

        # time step
        t = tf.get_variable(name='timestep', shape=[], initializer=tf.constant_initializer(0))

        def adam_fn(loss=loss, timestep=t, conv_layer_list = conv_layer_list, fc_layer_list = fc_layer_list):

            new_t = timestep.assign(timestep + 1)

            conv_wgrad_fisher_gradient = []
            fc_wgrad_fisher_gradient = []

            # get list of conv layer binarized wts, biases, here "wgrad" refers to either the weights or the binarized weights
            conv_layer_wgrad_list = []
            conv_layer_bias_list = []


            for convlayer in conv_layer_list:
                conv_layer_bias_list.append(convlayer.bias)
                if convlayer.binary:
                    conv_layer_wgrad_list.append(convlayer.wb)
                else:
                    conv_layer_wgrad_list.append(convlayer.weight)


                if convlayer.fisher and convlayer.is_binary:
                    conv_wgrad_fisher_gradient.append(gamma *
                        convlayer.fisherconst * 2.0 * gradients_impl._hessian_vector_product(loss, [convlayer.wb], [convlayer.perturbation ]))
                elif convlayer.fisher:
                    conv_wgrad_fisher_gradient.append(gamma *
                        convlayer.fisherconst * 2.0 * gradients_impl._hessian_vector_product(loss, [convlayer.weight], [convlayer.perturbation ]))
                else:
                    conv_wgrad_fisher_gradient.append(0.)
            # get list of fc layer binarized wts, biases
            fc_layer_wgrad_list = []
            fc_layer_bias_list = []
            # for fclayer in fc_layer_list:
            #     fc_layer_wgrad_list.append(fclayer.wb)
            #     fc_layer_bias_list.append(fclayer.bias)

            for fclayer in fc_layer_list:
                fc_layer_bias_list.append(fclayer.bias)
                if fclayer.binary:
                    fc_layer_wgrad_list.append(fclayer.wb)
                else:
                    fc_layer_wgrad_list.append(fclayer.weight)

                if fclayer.fisher and fclayer.binary:
                    fc_wgrad_fisher_gradient.append(gamma *
                        fclayer.fisherconst * 2.0 * gradients_impl._hessian_vector_product(loss, [fclayer.wb],
                                                                                             [fclayer.perturbation]))
                elif fclayer.fisher:
                    fc_wgrad_fisher_gradient.append(gamma *
                        fclayer.fisherconst * 2.0 * gradients_impl._hessian_vector_product(loss, [fclayer.weight],
                                                                                             [fclayer.perturbation]))
                else:
                    fc_wgrad_fisher_gradient.append(0.)

            print(fc_layer_wgrad_list)
            print(len(fc_layer_wgrad_list))
            # exit()
            # Calculate gradients wrt conv layer wb
            conv_layer_wgrad_grads = tf.gradients(loss, conv_layer_wgrad_list)
            # Calculate gradients wrt fc layer wb
            fc_layer_wgrad_grads = tf.gradients(loss, fc_layer_wgrad_list)
            # Calculate gradients wrt conv layer wb
            conv_layer_bias_grads = tf.gradients(loss, conv_layer_bias_list)
            # Calculate gradients wrt fc layer wb
            fc_layer_bias_grads = tf.gradients(loss, fc_layer_bias_list)

            conv_layer_w_gradient_tot = []
            fc_layer_w_gradient_tot = []

            for conv_w_grad, conv_w_fisher_grad in zip(conv_layer_wgrad_grads, conv_wgrad_fisher_gradient):
                conv_layer_w_gradient_tot.append(conv_w_grad + conv_w_fisher_grad)

            for fc_w_grad, fc_w_fisher_grad in zip(fc_layer_wgrad_grads, fc_wgrad_fisher_gradient):
                # print(fc_w_grad)
                # print(fc_w_fisher_grad)
                fc_layer_w_gradient_tot.append(fc_w_grad + fc_w_fisher_grad)

            # FOR CONV LAYERS:
            new_conv_m_wgrad = []
            new_conv_v_wgrad = []

            new_conv_m_b = []
            new_conv_v_b = []

            new_fc_m_wgrad  = []
            new_fc_v_wgrad = []

            new_fc_m_b = []
            new_fc_v_b = []

            #Calculate m, and v from adam for the wts
            for grad, layer in zip(conv_layer_w_gradient_tot, conv_layer_list):
                new_conv_m_wgrad.append( layer.m_w.assign(tf.squeeze(beta1 * layer.m_w + (1 - beta1) * grad)))
                new_conv_v_wgrad.append( layer.v_w.assign(tf.squeeze(beta2 * layer.v_w + (1 - beta2) * grad ** 2)))

            #Calculate m, and v from adam for the biases
            for grad, layer in zip(conv_layer_bias_grads, conv_layer_list):
                new_conv_m_b.append( layer.m_b.assign(beta1 * layer.m_b + (1 - beta1) * grad))
                new_conv_v_b.append( layer.v_b.assign(beta2 * layer.v_b + (1 - beta2) * grad ** 2))

            #FOR FC LAYERS:
            for grad, layer in zip(fc_layer_w_gradient_tot, fc_layer_list):
                new_fc_m_wgrad.append( layer.m_w.assign(tf.squeeze(beta1 * layer.m_w + (1 - beta1) * grad)))
                new_fc_v_wgrad.append( layer.v_w.assign(tf.squeeze(beta2 * layer.v_w + (1 - beta2) * grad ** 2)))

            #Calculate m, and v from adam for the biases
            for grad, layer in zip(fc_layer_bias_grads, fc_layer_list):
                new_fc_m_b.append( layer.m_b.assign(beta1 * layer.m_b + (1 - beta1) * grad))
                new_fc_v_b.append( layer.v_b.assign(beta2 * layer.v_b + (1 - beta2) * grad ** 2))

            #CALCULATE UPDATES:
            conv_updates_wgrad = []
            conv_updates_bias = []
            fc_updates_wgrad = []
            fc_updates_bias = []

            #For Conv layers wts
            for m, v in zip(new_conv_m_wgrad, new_conv_v_wgrad):
                conv_updates_wgrad.append(m / (tf.sqrt(v) + epsilon))

            #For conv layer bias
            for m, v in zip(new_conv_m_b, new_conv_v_b):
                conv_updates_bias.append(m / (tf.sqrt(v) + epsilon))


            #For FC layer wts
            for m, v in zip(new_fc_m_wgrad, new_fc_v_wgrad):
                fc_updates_wgrad.append(m / (tf.sqrt(v) + epsilon))

            #For FC layer bias
            for m, v in zip(new_fc_m_b, new_fc_v_b):
                fc_updates_bias.append(m / (tf.sqrt(v) + epsilon))

            return conv_updates_wgrad, conv_updates_bias, fc_updates_wgrad, fc_updates_bias, new_t


        # update = 0 in validation/test phase.
        nconv = len(conv_layer_list)
        nfc = len(fc_layer_list)
        def false_fn(t=t, n_conv=nconv, n_fc=nfc):

            convwzeros = []
            fcwzeros = []

            for i in range(n_conv):
                convwzeros.append(0.)

            for i in range(n_fc):
                fcwzeros.append(0.)

            return convwzeros,convwzeros, fcwzeros, fcwzeros, t


        # if is_training, do update
        conv_updates_w, conv_updates_b, fc_updates_w, fc_updates_b, new_t = tf.cond(is_training, adam_fn, false_fn)

        # adjust learning rate with beta
        lr = learning_rate * tf.sqrt(1 - beta2 ** new_t) / (1 - beta1 ** new_t)
        tf.summary.scalar('learning_rate', lr)

        # if is_binary, clip the weights to [-1, +1] before assignment
        # if is_binary:

        new_conv_w = []
        new_conv_b = []

        new_fc_w = []
        new_fc_b = []

        i=0

        conv_layer_bias_list=[]
        conv_layer_w_list=[]
        for convlayer in conv_layer_list:
            conv_layer_bias_list.append(convlayer.bias)
            conv_layer_w_list.append(convlayer.weight)

        fc_layer_w_list = []
        fc_layer_bias_list = []
        # for fclayer in fc_layer_list:
        #     fc_layer_wgrad_list.append(fclayer.wb)
        #     fc_layer_bias_list.append(fclayer.bias)


        for fclayer in fc_layer_list:
            fc_layer_bias_list.append(fclayer.bias)
            fc_layer_w_list.append(fclayer.weight)
        print('list of weights to be updated')
        print(fc_layer_w_list)
        print(len(fc_layer_w_list))

        for w, update in zip(conv_layer_w_list, conv_updates_w):
            print('conv layer index:'+str(i))

            if is_binary[i]:
                new_conv_w.append(w.assign(
                    tf.contrib.keras.backend.clip(w - lr * update, -1., 1.)
                ))
            else:
                new_conv_w.append(w.assign(w - lr * update))
            i+=1

        for w, update in zip(fc_layer_w_list, fc_updates_w):
            print('FC layer index:'+str(i))
            if is_binary[i]:
                new_fc_w.append(w.assign(
                    tf.contrib.keras.backend.clip(w - lr * update, -1., 1.)
                ))
            else:
                print(update)
                print(w)
                new_fc_w.append(w.assign(w - lr * update))
                # exit()

            i+=1
        # else:

            # new_conv_w = []
            # new_conv_b = []

            # new_fc_w = []
            # new_fc_b = []

            # for conv_layer, update in zip(conv_layer_list, conv_updates_w):
            #     new_conv_w.append(conv_layer.weight.assign(conv_layer.weight - lr * update))



            # for fc_layer, update in zip(fc_layer_list, fc_updates_w):
            #     new_fc_w.append(fc_layer.weight.assign(fc_layer.weight - lr * update))


        for conv_layer, update in zip(conv_layer_list, conv_updates_b):
            new_conv_b.append(conv_layer.bias.assign(conv_layer.bias - lr * update))
        for fc_layer, update in zip(fc_layer_list, fc_updates_b):
            new_fc_b.append(fc_layer.bias.assign(fc_layer.bias - lr * update))

        _updates = new_conv_w + new_conv_b + new_fc_w + new_fc_b
        _updates = tuple(_updates)

    return _updates










