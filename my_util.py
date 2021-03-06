import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
fdebug = open('debug.txt','w')
sess = tf.Session()
image_tf = tf.image.rgb_to_grayscale(tf.image.decode_png(tf.read_file('pic.png','r')))
image_PIL = tf.convert_to_tensor(np.asarray(Image.open('pic.png','r').convert('L')))
arr_tf = sess.run(tf.reshape(image_tf,[30,40]))
arr_PIL = sess.run(tf.reshape(image_PIL,[30,40]))
for i in range(30):
    for j in range(40):
        if arr_tf[i][j] != arr_PIL[i][j]:
            fdebug.write('i:'+str(i) + ' j:' + str(j) + '\n')
            fdebug.write('tf:' + str(arr_tf[i][j]) + ' PIL:' + str(arr_PIL[i][j]) + '\n')

'''
fdebug = open('debug.txt','w')
image = Image.open('ct1.tif').convert('L')
image_array = np.asarray(image)
image_tensor = tf.convert_to_tensor(image_array)
image_range1 = tf.to_float(tf.reshape(image_tensor,[64,64]))/tf.constant(255.)
sess = tf.Session()
arr = sess.run(image_range1)
Image.fromarray(sess.run(image_tensor)).save('ct1_8.png')
'''

'''
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
X_mb, _ = mnist.train.next_batch(2048)

image = Image.open('pic.png')
image_array = np.asarray(image)
fake = tf.read_file('fake.png','r')
image_tensor = tf.convert_to_tensor(image_array)
image_tensor = tf.image.rgb_to_grayscale(tf.image.decode_png(tf.read_file('pic.png','r')))
fake_tensor = tf.image.decode_png(fake)
#image_tensor = tf.reshape(tf.image.rgb_to_grayscale(image_tensor),[30,40])
shape = tf.shape(image_tensor)
image_tensor2 = tf.to_float(image_tensor)/tf.constant(255.)
image_tensor4 = tf.cast(tf.to_int32(image_tensor2*tf.constant(255.)), dtype=tf.uint8)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tf.shape(X_mb)))
print(sess.run(tf.shape(image_tensor4)))
print(sess.run(shape))
print(image_tensor)

Image.fromarray(sess.run(image_tensor4[:,:,0]), mode='L').save('new.png')

plt.figure(figsize=(0.4, 0.3), dpi=100)
plt.imshow(sess.run(image_tensor))
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.savefig('fake.png', dpi=100)

sess = tf.Session()
for i in range(512):
    image = tf.convert_to_tensor(X_mb[i])
    image = image * tf.constant(255.)
    image = tf.to_int32(image)
    image = tf.cast(image,dtype=tf.uint8)
    Image.fromarray(sess.run(image).reshape(28, 28)).convert('L').save('minist/minist_{}.png'.format(i))
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
    images = [whole_data[i] for i in index]
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