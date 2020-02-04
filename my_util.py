import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
'''
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
X_mb, _ = mnist.train.next_batch(2)

image = tf.read_file('pic.png','r')
image_tensor = tf.image.decode_png(image)
image_tensor = tf.reshape(tf.image.rgb_to_grayscale(image_tensor),[30,40])
shape = tf.shape(image_tensor)
image_tensor2 = tf.to_float(image_tensor)/tf.constant(255.)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tf.shape(X_mb)))
print(sess.run(image))
print(sess.run(shape))
print(image_tensor)
for i in range(30):
    for j in range(40):
        #continue
        #print(sess.run(tf.convert_to_tensor(X_mb[1])).reshape(28,28)[i][j])
        print(sess.run(image_tensor2[i][j]))
plt.figure(figsize=(4, 3), dpi=10)
plt.imshow(sess.run(image_tensor).reshape(30, 40), cmap='Greys_r')
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.savefig('fake.png', dpi=10)

plt.figure(figsize=(2.8, 2.8), dpi=10)
plt.imshow(sess.run(tf.convert_to_tensor(X_mb[1])).reshape(28, 28), cmap='Greys_r')
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.savefig('fake_minist.png', dpi=10)
'''


def readpic(filelist,batch_size):
    file_queue = tf.train.string_input_producer(filelist)

    reader = tf.WholeFileReader()

    key, value = reader.read(file_queue)

    #print(value)

    image = tf.image.decode_png(value)

    #print(image)

    image_resize = tf.image.resize_images(image,[480,640])

    image_resize.set_shape([480,640,3])

    #print(image_resize)

    image_batch = tf.train.batch([image_resize],batch_size=batch_size,num_threads=1,capacity=10000)

    #print(image_batch)
    return image_batch


def next_batch(whole_data, data_size, batch_size):
    index = np.random.randint(data_size, size=batch_size)
    images = [tf.to_float(tf.reshape(whole_data[i],[30,40]))/tf.constant(255.) for i in index]
    return images

def next_batch_line(whole_data, data_size, batch_size):
    index = np.random.randint(data_size, size=batch_size)
    images = [whole_data[i] for i in index]
    images_line = [tf.reshape(image, [-1]) for image in images]
    return images_line

'''
file_name = os.listdir("./data_DiscCenter_enhance/")

filelist = [os.path.join("./data_DiscCenter_enhance/", file) for file in file_name]

index = np.random.randint(len(filelist), size=8)

images = [tf.image.decode_png(tf.read_file(filelist[i], 'r')) for i in index]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.shape(tf.convert_to_tensor(images))))
'''

#print(np.random.uniform(-1., 1., size=[2, 3, 4, 3]))