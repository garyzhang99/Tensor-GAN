import tensorflow as tf
import os
import numpy as np

width = 128
height = 128
classes = 2
lr = 1e-3
batch_size = 8
train_portion = 0.3
train_iter = 200000

def loadData():
    label0files = os.listdir('label_0_denoised')
    label0files = [os.path.join('label_0_denoised/',file) for file in label0files]
    label1files = os.listdir('label_1_denoised')
    label1files = [os.path.join('label_1_denoised/', file) for file in label1files]
    #label0files = label0files[:20]
    #label1files = label1files[:20]
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


X = tf.placeholder(tf.float32,[None,height,width],name="X")
Y = tf.placeholder(tf.int32,[None],name="Y")
drop_out = tf.placeholder(tf.float32,name="drop_out")

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
        pre,l = sess.run([prediction,loss],feed_dict={X:[testImages[i]],Y:[testLabels[i]],drop_out:0.1})
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
        sess.run(optimizer,feed_dict={X:images,Y:labels,drop_out:0.1})
        #print(l)
        #print(git)
        #print(hot)
        if _ % 10000 == 0:
            f1,tp,fp,tn,fn,testloss = test(test_images,testLabels,sess)
            record.write('f1:'+str(f1))
            record.write(' tp,fp,tn,fn:'+str(tp) + ' ' + str(fp) + ' ' + str(tn) + ' ' + str(fn)+'\n')

trainFiles,trainLabels,testFiles,testLabels = loadData()
train(trainFiles,trainLabels,testFiles,testLabels)

