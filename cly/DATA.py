import os
import struct
import numpy as np

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

X_train,y_train = load_mnist("E:/2018Fall/CPE695/final/dataset/")
X_test,y_test = load_mnist("E:/2018Fall/CPE695/final/dataset/",kind="t10k")

np.savetxt('train_set.csv', X_train,
           fmt='%i', delimiter=',')
np.savetxt('train_label.csv', y_train,
           fmt='%i', delimiter=',')
np.savetxt('test_set.csv', X_test,
           fmt='%i', delimiter=',')
np.savetxt('test_label.csv', y_test,
           fmt='%i', delimiter=',')