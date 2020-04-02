import tensorflow as tf
import os
import numpy as np
import random

width = 128
height = 128
classes = 2
lr = 1e-4
batch_size = 16
train_portion = 0.8
train_iter = 50000
drop_rate = 0.2

X = tf.placeholder(tf.float32,[None,height,width,1],name="X")
Y = tf.placeholder(tf.int32,[None],name="Y")
drop_out = tf.placeholder(tf.float32,name="drop_out")

def image_enhence(x):
    return tf.image.random_flip_left_right(x)
    #return tf.add(x,tf.random_normal(shape=tf.shape(x),mean=0,stddev=0.05,dtype=tf.float32))


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


def max_pool(x,shape):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=shape, strides=[1, 2, 2, 1], padding='SAME')


x = tf.reshape(X, [-1, height, width, 1])
W_conv1 = weight_variable([4, 4, 1, 32])
b_conv1 = bias_variable([32])
conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 2, 2, 1],padding='SAME')+b_conv1)#63*63*32
pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1, 2, 2, 1],padding='SAME')#32*32*32
#h_conv1 = tf.nn.relu(conv2d(tf.reshape(x,[-1,height,width,1]), W_conv1) + b_conv1)
#h_pool1 = max_pool(h_conv1,[1,2,2,1])  # output size 64x64x32

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides=[1, 2, 2, 1],padding='SAME')+b_conv2)#16*16*64
pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1, 2, 2, 1],padding='SAME')#8*8*64

W_conv3 = weight_variable([2, 2, 64, 128])
b_conv3 = bias_variable([128])
conv3 = tf.nn.relu(tf.nn.conv2d(pool2, W_conv3, strides=[1,2,2,1], padding='SAME')+b_conv3)#4*4*128

W_dense1 = weight_variable([4 * 4 * 128, 256])
b_dense1 = bias_variable([256])
flat1 = tf.nn.sigmoid(tf.matmul(tf.reshape(conv3,[-1,4*4*128]),W_dense1)+b_dense1)
drop1 = tf.nn.dropout(flat1,drop_out)

W_dense2 = weight_variable([256, 64])
b_dense2 = bias_variable([64])
flat2 = tf.nn.sigmoid(tf.matmul(drop1,W_dense2)+b_dense2)
drop2 = tf.nn.dropout(flat2,drop_out)

W_dense3 = weight_variable([64, 2])
b_dense3 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(drop2,W_dense3)+b_dense3)

#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 64x64x64
#h_pool2 = max_pool_2x2(h_conv2)  # 32x32x64

#W_fc1 = weight_variable([32 * 32 * 64, 1024])
#b_fc1 = bias_variable([1024])

#h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 32 * 64])
#h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#h_fc1_drop = tf.nn.dropout(h_fc1, drop_out)

#W_fc2 = weight_variable([1024, 2])
#b_fc2 = bias_variable([2])
#prediction2 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)#sigmoid

cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(Y,classes) * tf.log(prediction + 1e-6), reduction_indices=[1]))
# 由于 prediction 可能为 0， 导致 log 出错，最后结果会出现 NA
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
optimizer2 = tf.train.AdamOptimizer(lr).minimize(cross_entropy)






def loadData():
    '''
    auxfiles = os.listdir('auxdata/')
    auxfiles = [os.path.join('auxdata',file) for file in auxfiles]
    label0files = os.listdir('label_0_denoised')
    label0files = [os.path.join('label_0_denoised/',file) for file in label0files]
    label1files = os.listdir('label_1_denoised')
    label1files = [os.path.join('label_1_denoised/', file) for file in label1files]
    #label0files = label0files[:len(label1files)]
    #label1files = label1files[:20]
    index0 = random.sample(range(len(label0files)), int(len(label0files) * train_portion))
    index1 = random.sample(range(len(label1files)), int(len(label1files) * train_portion))
    indexAux = random.sample(range(len(auxfiles)), int((len(label0files)-len(label1files))*train_portion))
    trainFiles = []
    trainLabels = []
    testFiles = []
    testLabels = []
    print('total0,train0: ' + str(len(label0files)) + " " + str(len(index0)))
    print('total1,train1: ' + str(len(label1files)) + " " + str(len(index1)))
    print('aux1: ' + str(len(indexAux)))
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
    for i in range(len(auxfiles)):
        if i in indexAux:
            trainFiles.append(auxfiles[i])
            trainLabels.append(1)
    '''
    trainFiles = []
    trainLabels = []
    testFiles = []
    testLabels = []
    train0Files = ['train0/'+fname for fname in os.listdir('train0')]
    train1Files = ['train1/'+fname for fname in os.listdir('train1')]
    test0Files = ['test0/'+fname for fname in os.listdir('test0')]
    test1Files = ['test1/'+fname for fname in os.listdir('test1')]
    #train0Files = train0Files[:20]
    #train1Files = train1Files[:16]
    auxfiles = os.listdir('auxdata/')
    auxfiles = [os.path.join('auxdata',file) for file in auxfiles]
    #indexAux = random.sample(range(len(auxfiles)), (len(train0Files)-len(train1Files))*2)
    #train0Files = train0Files[:len(train1Files)]
    for f in train0Files:
        trainFiles.append(f)
        trainLabels.append(0)
    for f in train1Files:
        trainFiles.append(f)
        trainLabels.append(1)
    for f in test0Files:
        testFiles.append(f)
        testLabels.append(0)
    for f in test1Files:
        testFiles.append(f)
        testLabels.append(1)
    for i in range(len(auxfiles)):
        #if i in indexAux:
        trainFiles.append(auxfiles[i])
        trainLabels.append(1)
    print('train0:' + str(len(train0Files)))
    print('train1:'+str(len(train1Files)))
    print('test0:'+str(len(test0Files)))
    print('test1:'+str(len(test1Files)))
    print('aux:'+str(len(auxfiles)))

    return trainFiles,trainLabels,testFiles,testLabels
def test(testImages,testLabels,sess):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    testLoss = 0

    for i in range(len(testImages)):
        pre,l = sess.run([tf.argmax(prediction,1),cross_entropy],feed_dict={X:[testImages[i]],Y:[testLabels[i]],drop_out:drop_rate})
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
    elif TP == 0:
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
    train_data = []
    c = 0
    for fpath in trainFiles:
        if fpath[0]=='a':
            train_data.append(tf.image.decode_png(tf.read_file(fpath,'r')))
            c+=1
        else:
            train_data.append(tf.image.rgb_to_grayscale(tf.image.decode_png(tf.read_file(fpath,'r'))))
    print('total aux:' + str(c))
    #train_data = [tf.image.rgb_to_grayscale(tf.image.decode_png(tf.read_file(file, 'r'))) for file in trainFiles]
    train_images = [tf.to_float(tf.reshape(data, [height, width])) / tf.constant(255.) for data in train_data]
    train_images = [tf.reshape(image, [height,width,1]).eval(session=sess) for image in train_images]

    test_data = [tf.image.rgb_to_grayscale(tf.image.decode_png(tf.read_file(file, 'r'))) for file in testFiles]
    test_images = [tf.to_float(tf.reshape(data, [height, width])) / tf.constant(255.) for data in test_data]
    test_images = [tf.reshape(image, [height, width,1]).eval(session=sess) for image in test_images]

    print('data loaded')
    record = open('result.txt','a+')
    for _ in range(train_iter):
        index = np.random.randint(train_size, size=batch_size)
        images = [train_images[i] for i in index]
        labels = [trainLabels[i] for i in index]
        closs,o2 = sess.run([cross_entropy,optimizer2],feed_dict={X:images,Y:labels,drop_out:drop_rate})
        #print(l)
        #print(git)
        #print(hot)
        if _ % 1000 == 0:
            f1,tp,fp,tn,fn,testloss = test(test_images,testLabels,sess)
            print('f1: ' + str(f1))
            print(' tp,fp,tn,fn: '+str(tp) + ' ' + str(fp) + ' ' + str(tn) + ' ' + str(fn))
            print('loss: ' + str(testloss))
            print('closs: ' + str(closs))
            record.write('f1:'+str(f1))
            record.write(' tp,fp,tn,fn:'+str(tp) + ' ' + str(fp) + ' ' + str(tn) + ' ' + str(fn)+'\n')
            record.write('loss: ' + str(testloss))
            record.write('closs:" ' + str(closs) + '\n')

trainFiles,trainLabels,testFiles,testLabels = loadData()
train(trainFiles,trainLabels,testFiles,testLabels)
