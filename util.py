import numpy as np
import tensorflow as tf

weight_std = 0.2 #正态分布标准差


def next_batch(num, data):
    '''
    Return a random batch of samples. batch size is num,
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]

    return np.asarray(data_shuffle)

def normal_init(size):
    return tf.random_normal(shape = size, stddev = weight_std)
    #size是张量的形状

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
    #泽维尔初始化

def get_number_parameters(vars):
	total = 0
	for var in vars:
		total += np.prod(var.get_shape().as_list()) #var的元素数量
	return total