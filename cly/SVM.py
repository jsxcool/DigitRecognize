from sklearn.svm import SVC
import os
import struct
import numpy as np
import time
import random

def load_mnist(path, kind="train"):
    """Load MNIST data from `path`"""
    labels_path = path+"%s-labels.idx1-ubyte"% kind
    images_path = path+"%s-images.idx3-ubyte"% kind
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

X_train,Y_train = load_mnist("E:/2018Fall/CPE695/final/dataset/")
X_test,y_test = load_mnist("E:/2018Fall/CPE695/final/dataset/",kind="t10k")

# create model

# 对10个数字进行分类测试
def test_SVM():
    clf = SVC(C=100.0, kernel='poly', gamma=0.03)
    #clf.fit(X_train,y_train)
    tcName = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    tst = time.clock()
    allErrCount = 0
    allErrorRate = 0.0
    allScore = 0.0
    for tcn  in tcName:
        indx = np.where(Y_train == int(tcn))
        sample1 = random.sample(range(0,len(indx)),len(indx)/2)
        train_index = indx[sample1]
        x_train,y_train = X_train[train_index],Y_train[train_index]
        clf.fix(x_train,y_train)
        index = np.where(y_test == int(tcn))
        sample2 = random.sample(range(0, len(index)), 100)
        test_index = index[sample2]
        tdataMat = X_test[test_index]
        tdataLabel = y_test[test_index]
        print("test dataMat shape: ",tdataMat.shape,"test dataLabel len: ",len(tdataLabel))
        #print("test dataLabel: {}".format(len(tdataLabel)))
        pre_st = time.clock()
        preResult = clf.predict(tdataMat)
        pre_et = time.clock()
        print("Recognition",tcn,"spent",'%.4fs' % (pre_et-pre_st))
        #print("predict result: {}".format(len(preResult)))
        errCount = len([x for x in preResult if x!=int(tcn)])
        print("errorCount:",errCount)
        allErrCount += errCount
        score_st = time.clock()
        score = clf.score(tdataMat, tdataLabel)
        score_et = time.clock()
        print("computing score spent",'%.4fs' %(score_et-score_st))
        allScore += score
        print("score:",'%.6f' %score)
        print("error rate is",'%.6f' %(1-score))
        print("---------------------------------------------------------")


    tet = time.clock()
    print("Testing All class total spent",'%.6fs' %(tet-tst))
    print("All error Count is:",allErrCount)
    avgAccuracy = allScore/10.0
    print("Average accuracy is:",'%.6f' %avgAccuracy)
    print("Average error rate is:",'%.6f' % (1-avgAccuracy))

if __name__ == "__main__":
    test_SVM()