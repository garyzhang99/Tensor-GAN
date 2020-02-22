import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import my_util
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

data_path = "./1st_followup_png/"

output_path = "./1st_followup_png_out/"

file_name = os.listdir(data_path)

filelist = [os.path.join(data_path, file) for file in file_name]

data_size = len(filelist)

whole_data = [tf.image.rgb_to_grayscale(tf.image.decode_png(tf.read_file(file,'r'))) for file in filelist]

whole_images =[tf.to_float(tf.reshape(data,[256,256]))/tf.constant(255.) for data in whole_data]

tf.reset_default_graph()

save_order = 0

batch_size = 64

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 256 * 256])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 32

reshaped_dim = [-1, 15, 15, dec_in_channels]
inputs_decoder = 225 * dec_in_channels / 2
print(inputs_decoder)

def picsave(sample,sess):
    global save_order
    tensor255 = sample * tf.constant(255.)
    tensorint32 = tf.to_int32(tensor255)
    image_tensor = tf.cast(tensorint32, dtype=tf.uint8)
    Image.fromarray(sess.run(image_tensor), mode='L').save(output_path+'{}.png'.format(str(save_order)))
    save_order += 1

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, 256, 256, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=8, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=8, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=8, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(sd))

        return z, mn, sd


def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder * 2, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=256 * 256, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 256, 256])
        return img

sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)

unreshaped = tf.reshape(dec, [-1, 256*256])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
'''
print('test')
x1 = tf.layers.dense(sampled, units=inputs_decoder, activation=lrelu)
x2 = tf.layers.dense(x1, units=inputs_decoder * 2 + 1, activation=lrelu)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


batch, _ = mnist.train.next_batch(batch_size)

print(
    sess.run(tf.shape(sampled),feed_dict={X_in: batch.reshape(batch_size, 28, 28),
                                      Y: batch.reshape(batch_size, 28, 28), keep_prob: 1.0})
)
print(
    sess.run(tf.shape(x1),feed_dict={X_in: batch.reshape(batch_size, 28, 28),
                                      Y: batch.reshape(batch_size, 28, 28), keep_prob: 1.0})
)
print(
    sess.run(tf.shape(x2),feed_dict={X_in: batch.reshape(batch_size, 28, 28),
                                      Y: batch.reshape(batch_size, 28, 28), keep_prob: 1.0})
)

'''
print('begin')

if not os.path.exists(output_path):
    os.mkdir(output_path)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(40000):
    print(i)
    batch = my_util.next_batch(whole_data=whole_images,data_size=len(whole_images),batch_size=batch_size)
    #batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
    sess.run(optimizer, feed_dict={X_in: batch.reshape(batch_size, 256, 256),
                                      Y: batch.reshape(batch_size, 256, 256), keep_prob: 0.8})

    if i % 1000 == 0:
        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd],
                                               feed_dict={X_in: batch.reshape(batch_size, 256, 256),
                                      Y: batch.reshape(batch_size, 256, 256), keep_prob: 1.0})

        randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
        imgs = sess.run(dec, feed_dict={sampled: randoms, keep_prob: 1.0})
        for i in range(len(imgs)):
            picsave(imgs[i],sess)
        print(i, ls, np.mean(i_ls), np.mean(d_ls))

