#https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import my_util
import matplotlib.gridspec as gridspec
from PIL import Image
from util import get_number_parameters

width = 256

height = 256

tunnel = 1

data_path = "./1st_followup_png/"

output_path = "./1st_followup_png_out_gan/"

file_name = os.listdir(data_path)

filelist = [os.path.join(data_path, file) for file in file_name]

data_size = len(filelist)

whole_data = [tf.image.rgb_to_grayscale(tf.image.decode_png(tf.read_file(file,'r'))) for file in filelist]

whole_images =[tf.to_float(tf.reshape(data,[height,width]))/tf.constant(255.) for data in whole_data]
save_order = 0

pixel = width*height

batch_size = 128

z_dim = 1000

h_dim = 4096

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, pixel])

D_W1 = tf.Variable(xavier_init([pixel, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


Z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, pixel]))
G_b2 = tf.Variable(tf.zeros(shape=[pixel]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

print("Total number of parameters: {}".format(get_number_parameters(theta_G+theta_D)))

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(sample):
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
    plt.imshow(sample.reshape(480, 640, 3))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    return fig

def picsave(sample,sess):
    global save_order
    tensor255 = sample * tf.constant(255.)
    tensorint32 = tf.to_int32(tensor255)
    image_tensor = tf.cast(tensorint32, dtype=tf.uint8)
    Image.fromarray(sess.run(image_tensor), mode='L').save(output_path+'{}.png'.format(str(save_order)))
    save_order += 1

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

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
whole_images = [tf.reshape(image,[-1]).eval(session=sess) for image in whole_images]

if not os.path.exists(output_path):
    os.makedirs(output_path)

i = 0

for it in range(1000000):
    if it % 200 == 0:
        print(it)
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(8, z_dim)})
        for i in range(8):
            picsave(tf.reshape(samples[i],[height,width]),sess)

    X_mb = my_util.next_batch(whole_data=whole_images, data_size=data_size, batch_size=batch_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, z_dim)})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()