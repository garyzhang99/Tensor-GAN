import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tnsr import *

import tensorflow as tf
from util import normal_init, get_number_parameters
import my_util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

data_path = "./label0_data_copy/"

output_path = "./eye_out_label0_copy/"

file_name = os.listdir(data_path)

filelist = [os.path.join(data_path, file) for file in file_name]

data_size = len(filelist)

whole_data = [tf.image.rgb_to_grayscale(tf.image.decode_png(tf.read_file(file, 'r')))
              for file in filelist]

batch_size = 32

lr = 5e-3

core_h_dim = 36
core_w_dim = 48

gen_h_dim = 36
gen_w_dim = 48


X = tf.placeholder(tf.float32, shape=[None, 30, 40])

# Weight matrices and bias tensor for the first and second tensor layer for the discriminator D
D_U_00 = tf.Variable(normal_init([core_h_dim, 30]))
D_U_01 = tf.Variable(normal_init([core_w_dim, 40]))
#D_U_02 = tf.Variable(normal_init([3, 3]))
D_b_0 = tf.Variable(tf.zeros(shape=[core_h_dim, core_w_dim]))

D_U_10 = tf.Variable(normal_init([1, core_h_dim]))
D_U_11 = tf.Variable(normal_init([1, core_w_dim]))
#D_U_12 = tf.Variable(normal_init([1, 3]))
D_b_1 = tf.Variable(tf.zeros(shape=[1,1]))

# Parameters for discriminator
theta_D = [D_U_00, D_U_01, D_U_10, D_U_11, D_b_0, D_b_1]


# Prior Z
Z = tf.placeholder(tf.float32, shape=[None, gen_h_dim, gen_w_dim])

# Weight matrices and bias tensor forthe first and second tensor layer for the generator G
G_U_00 = tf.Variable(normal_init([core_h_dim, gen_h_dim]))
G_U_01 = tf.Variable(normal_init([core_w_dim, gen_w_dim]))
#G_U_02 = tf.Variable(normal_init([3, 3]))
G_b_0 = tf.Variable(tf.zeros(shape=[core_h_dim, core_w_dim]))

G_U_10 = tf.Variable(normal_init([30, core_h_dim]))
G_U_11 = tf.Variable(normal_init([40, core_w_dim]))
#G_U_12 = tf.Variable(normal_init([3, 3]))
G_b_1 = tf.Variable(tf.zeros(shape=[30, 40]))

# Parameters for generator
theta_G = [G_U_00, G_U_01, G_U_10, G_U_11, G_b_0, G_b_1]

def sample_z(shape):
    arr = np.random.uniform(-1., 1., size=shape)
    #print(arr.shape)
    return arr


def generator(z):

    # First Tensorized layer
    out = tensor_layer(z, [G_U_00, G_U_01], G_b_0, tf.nn.relu)

    # Second Tensorized layer
    out = tensor_layer(out, [G_U_10, G_U_11], G_b_1, tf.nn.sigmoid)

    return out


def discriminator(x):
    # First Tensorized layer
    out = tensor_layer(x, [D_U_00, D_U_01], D_b_0, tf.nn.relu)

    # Return the logit and prob reoresentation after sigmoid
    return tensor_layer(out, [D_U_10, D_U_11], D_b_1, tf.nn.sigmoid), \
           tensor_layer(out, [D_U_10, D_U_11], D_b_1, identity)



def plot(sample):
    fig = plt.figure(figsize=(0.4, 0.3), dpi=100)
    plt.imshow(sample.reshape(30, 40), cmap='Greys_r')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    return fig

print("Total number of parameters: {}".format(get_number_parameters(theta_G+theta_D)))

G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(G_loss, var_list=theta_G)

#mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(output_path):
    os.makedirs(output_path)

i = 0

for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_z([1, gen_h_dim, gen_w_dim])})
        fig = plot(samples[0])
        plt.savefig(output_path+'{}.png'.format(str(i)), dpi=100)
        i += 1
        plt.close(fig)

    X_mb = my_util.next_batch(whole_data=whole_data, data_size=data_size, batch_size=batch_size)
    X_mb = [tensor.eval(session=sess) for tensor in X_mb]

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_z([batch_size, gen_h_dim, gen_w_dim])})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_z([batch_size, gen_h_dim, gen_w_dim])})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print('')

