import tensorflow as tf
import os
import numpy as np

width = 128
height = 128
classes = 2
lr = 1e-3
batch_size = 2
train_portion = 0.3
train_iter = 200000

X = tf.placeholder(tf.float32,[None,height,width],name="X")
Y = tf.placeholder(tf.int32,[None],name="Y")
drop_out = tf.placeholder(tf.float32,name="drop_out")


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(tf.reshape(X,[-1,height,width,1]), W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # output size 64x64x32

W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 64x64x64
h_pool2 = max_pool_2x2(h_conv2)  # 32x32x64

W_fc1 = weight_variable([32 * 32 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 32 * 64])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, drop_out)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
prediction2 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)#sigmoid

cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(Y,classes) * tf.log(prediction2 + 1e-10), reduction_indices=[1]))
# 由于 prediction 可能为 0， 导致 log 出错，最后结果会出现 NA
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
optimizer2 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)






def loadData():
    label0files = os.listdir('label_0_denoised')
    label0files = [os.path.join('label_0_denoised/',file) for file in label0files]
    label1files = os.listdir('label_1_denoised')
    label1files = [os.path.join('label_1_denoised/', file) for file in label1files]
    label0files = label0files[:45]
    label1files = label1files[:40]
    index0 = np.random.randint(len(label0files), size=int(len(label0files) * train_portion))
    index1 = np.random.randint(len(label1files), size=int(len(label1files) * train_portion))
    trainFiles = []
    trainLabels = []
    testFiles = []
    testLabels = []
    for i in range(len(label0files)):
        if i in index0:
            trainFiles.append(label0files[i])
            trainLabels.append(0)
        else:
            testFiles.append(label0files[i])
            testLabels.append(0)
    for i in range(len(label1files)):
        if i in index1:
            trainFiles.append(label1files[i])
            trainLabels.append(1)
        else:
            testFiles.append(label1files[i])
            testLabels.append(1)
    return trainFiles,trainLabels,testFiles,testLabels


def CNN(x,drop_rate):
    x = tf.reshape(x,[-1,height,width,1])
    convolution1 = tf.layers.conv2d(inputs=x,filters=10,kernel_size=[4,4],activation=tf.nn.relu)
    pooling1 = tf.layers.average_pooling2d(convolution1,pool_size=[4,4],strides=[2,2])
    convolution2 = tf.layers.conv2d(inputs=pooling1,filters=40,kernel_size=[4,4],activation=tf.nn.relu)
    pooling2 = tf.layers.average_pooling2d(convolution2,pool_size=[2,2],strides=[2,2])
    flatten = tf.layers.flatten(pooling2)
    dense1 = tf.layers.dense(inputs=flatten,units=800,activation=tf.nn.relu)
    drop = tf.layers.dropout(inputs=dense1,rate=drop_rate)
    dense2 = tf.layers.dense(inputs=drop,units=classes,activation=tf.nn.sigmoid)
    return dense2

logits = CNN(X,drop_out)
prediction = tf.argmax(logits,1)
loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(Y,classes),logits=logits)
#loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

def test(testImages,testLabels,sess):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    testLoss = 0

    for i in range(len(testImages)):
        pre,l = sess.run([tf.arg_max(prediction2,1),cross_entropy],feed_dict={X:[testImages[i]],Y:[testLabels[i]],drop_out:0.1})
        testLoss+=l
        if pre == 1 and testLabels[i] == 1:
            TP += 1
        elif pre == 1 and testLabels[i] == 0:
            FP += 1
        elif pre == 0 and testLabels[i] == 0:
            TN += 1
        else:
            FN += 1
    if TP+FP == 0 or TP+FN == 0:
        f1 = 0
    else:
        precision = float(TP)/float(TP+FP)
        recall = float(TP)/float(TP+FN)
        f1 = 2*(precision*recall)/(precision+recall)
    return f1,TP,FP,TN,FN,testLoss/len(testImages)

def train(trainFiles,trainLabels,testFiles,testLabels):
    sess = tf.Session()
    print('initialize...')
    sess.run(tf.global_variables_initializer())
    print('initialized')

    train_size = len(trainFiles)
    train_data = [tf.image.rgb_to_grayscale(tf.image.decode_png(tf.read_file(file, 'r'))) for file in trainFiles]
    train_images = [tf.to_float(tf.reshape(data, [height, width])) / tf.constant(255.) for data in train_data]
    train_images = [tf.reshape(image, [height,width]).eval(session=sess) for image in train_images]

    test_data = [tf.image.rgb_to_grayscale(tf.image.decode_png(tf.read_file(file, 'r'))) for file in testFiles]
    test_images = [tf.to_float(tf.reshape(data, [height, width])) / tf.constant(255.) for data in test_data]
    test_images = [tf.reshape(image, [height, width]).eval(session=sess) for image in test_images]

    print('data loaded')
    record = open('result.txt','a+')
    for _ in range(train_iter):
        print(_)
        index = np.random.randint(train_size, size=batch_size)
        images = [train_images[i] for i in index]
        labels = [trainLabels[i] for i in index]
        closs,o2 = sess.run([cross_entropy,optimizer2],feed_dict={X:images,Y:labels,drop_out:0.1})
        #print(l)
        #print(git)
        #print(hot)
        if _ % 300 == 0:
            print('iter: ' + str(_))
            print('closs: ' + str(closs))
        if _ % 10000 == 0:
            f1,tp,fp,tn,fn,testloss = test(test_images,testLabels,sess)
            print('f1: ' + str(f1))
            print(' tp,fp,tn,fn: '+str(tp) + ' ' + str(fp) + ' ' + str(tn) + ' ' + str(fn))
            print('loss: ' + str(testloss))
            record.write('f1:'+str(f1))
            record.write(' tp,fp,tn,fn:'+str(tp) + ' ' + str(fp) + ' ' + str(tn) + ' ' + str(fn)+'\n')

trainFiles,trainLabels,testFiles,testLabels = loadData()
train(trainFiles,trainLabels,testFiles,testLabels)

