import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

'''
image = tf.read_file('29_DiscCenter.png','r')
image_tensor = tf.image.decode_png(image)
image_tensor = tf.reshape(tf.image.rgb_to_grayscale(image_tensor),[480,640])
shape = tf.shape(image_tensor)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(image))
print(sess.run(shape))
print(image_tensor)
print(sess.run(image_tensor))
plt.figure(figsize=(6.4, 4.8), dpi=100)
plt.imshow(sess.run(image_tensor).reshape(480, 640))
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.savefig('fake.png', dpi=100)
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
    images = [tf.reshape(whole_data[i],[480,640]) for i in index]
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