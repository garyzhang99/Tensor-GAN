import tensorflow as tf
import os
import math

width = 128
height = 128
classes = 2
lr = 1e-4
batch_size = 16
train_portion = 0.8
train_iter = 50000
drop_rate = 0.2
k = 2
tp = 0
fp = 0
tn = 0
fn = 0

xtrain = tf.placeholder(tf.float32,[None,height*width],name="xtrain")
xtest = tf.placeholder(tf.float32,[height*width],name="xtest")

def loaddata():
    trainFiles = []
    trainLabels = []
    testFiles = []
    testLabels = []
    train0Files = ['train0/' + fname for fname in os.listdir('train0')]
    train1Files = ['train1/' + fname for fname in os.listdir('train1')]
    test0Files = ['test0/' + fname for fname in os.listdir('test0')]
    test1Files = ['test1/' + fname for fname in os.listdir('test1')]
    # train0Files = train0Files[:20]
    # train1Files = train1Files[:16]
    auxfiles = os.listdir('auxdata/')
    auxfiles = [os.path.join('auxdata', file) for file in auxfiles]
    # indexAux = random.sample(range(len(auxfiles)), (len(train0Files)-len(train1Files))*2)
    # train0Files = train0Files[:len(train1Files)]
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
        # if i in indexAux:
        trainFiles.append(auxfiles[i])
        trainLabels.append(1)
    print('train0:' + str(len(train0Files)))
    print('train1:' + str(len(train1Files)))
    print('test0:' + str(len(test0Files)))
    print('test1:' + str(len(test1Files)))
    print('aux:' + str(len(auxfiles)))
    return trainFiles, trainLabels, testFiles, testLabels

def test(prelist,truelabel):
    pre = 0
    global tp,fp,tn,fn
    for prelabel in prelist:
        if prelabel > 0:
            pre+=1
        else:
            pre-=1
    if pre == 0:
        pre = prelist[0]

    if pre>0 and truelabel>0:
        tp+=1
    elif pre>0 and truelabel<=0:
        fp+=1
    elif pre<=0 and truelabel<=0:
        tn+=1
    else:
        fn+=1

distance = tf.reduce_sum(tf.abs(tf.add(xtrain, tf.negative(xtest))),axis=1)

indices = tf.nn.top_k(-distance,k).indices

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    trainFiles,trainLabels,testFiles,testLabels = loaddata()
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
    train_images = [tf.reshape(image,[height*width]).eval(session=sess) for image in train_images]

    test_data = [tf.image.rgb_to_grayscale(tf.image.decode_png(tf.read_file(file, 'r'))) for file in testFiles]
    test_images = [tf.to_float(tf.reshape(data, [height, width])) / tf.constant(255.) for data in test_data]
    test_images = [tf.reshape(image,[height*width]).eval(session=sess) for image in test_images]

    for i in range(len(test_images)):
        indis= sess.run(indices,feed_dict={xtrain:train_images,xtest:test_images[i]})
        prelist = [trainLabels[_] for _ in indis]
        test(prelist,testLabels[i])
        if i%100 == 0:
            print(i)

    if tp+fp == 0 or tp+fn == 0:
        f1 = 0
    elif tp == 0:
        f1 = 0
    else:
        precision = float(tp)/float(tp+fp)
        recall = float(tp)/float(tp+fn)
        f1 = 2*(precision*recall)/(precision+recall)
    print(tp,fp,tn,fn)
    print("f1: " + str(f1))
