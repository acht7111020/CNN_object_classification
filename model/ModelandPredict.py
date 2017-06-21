import numpy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import initializers

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.misc as misc
from keras.models import model_from_json

class VGGmodel(object):

    def __init__(self):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.model = Sequential()        
        data = sio.loadmat('caltech_del.mat')
        self.categories = data['categories']
        self.num_classed = len(self.categories)
        
    def load_model(self):
        # load json and create model
        json_file = open('caltech_del/model7.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("caltech_del/model7.h5")
        print("Loaded model from disk")
    
    def model_compile(self):
        epochs = 50
        adam = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        print(self.model.summary())
    
    def model_predict(self, figure):
        if len(figure.shape) == 2:
            figure = np.transpose(np.array([figure, figure, figure]), (2, 0, 1))
        test = misc.imresize(figure, (128,128,3))
        test = test.astype('float32')
        test[:,:,0] -= 123.68
        test[:,:,1] -= 116.78
        test[:,:,2] -= 103.94
        test = test.transpose(2,0,1)
        test = np.expand_dims(test, axis=0)
        y_proba = self.model.predict(test)
        classed = self.model.predict_classes(test, verbose=0)
        if max(y_proba[0]) > 0.5:
            return(self.categories[classed[0]])
        else:
            return ''
    
    def test(self):
        data = sio.loadmat('caltech_del.mat')
        X_test = data['X_test']
        y_test = data['y_test']
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

from keras.datasets import cifar10
class CIFARmodel(object):

    def __init__(self, num_classed= 10):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.model = Sequential()
        self.num_classed = num_classed
        self.categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
    def load_model(self):
        # load json and create model
        json_file = open('cifar10/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("cifar10/model.h5")
        print("Loaded model from disk")
    
    def model_compile(self):
        epochs = 50
        lrate = 0.01
        adam = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        print(self.model.summary())
    
    def model_predict(self, figure):
        test = misc.imresize(figure, (32,32,3)).transpose(2,0,1)
        test = test.astype('float32')
        test /= 255
        test = np.expand_dims(test, axis=0)
        y_proba = self.model.predict(test)
        classed = self.model.predict_classes(test, verbose=0)
        if max(y_proba[0]) > 0.5:
            return(self.categories[classed[0]])
        else:
            return ''
    
    def test(self):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_test = X_test.astype('float32')
        X_test = X_test / 255.0
        # one hot encode outputs
        y_test = np_utils.to_categorical(y_test)
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

from keras.datasets import cifar100
class CIFAR100model(object):

    def __init__(self, num_classed= 10):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.model = Sequential()
        self.num_classed = num_classed
        self.categories = ['apples', 'aquarium fish', 'baby', 'bear', 'beaver',
            'bed', 'bee', 'beetle', 'bicycle', 'bottles',
            'bowls','boy', 'bridge', 'bus', 'butterfly', 
            'camel', 'cans', 'castle', 'caterpillar', 'cattle', 
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 
            'couch', 'crab', 'crocodile', 'cups', 
            'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 
            'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer keyboard', 'lamp', 
            'lawn-mower', 'leopard', 'lion', 'lizard', 'lobster', 
            'man', 'maple', 'motorcycle', 'mountain', 'mouse', 
            'mushrooms', 'oak', 'oranges', 'orchids', 'otter', 
            'palm', 'pears', 'pickup truck', 'pine', 'plain', 
            'plates', 'poppies', 'porcupine', 'possum', 'rabbit', 
            'raccoon', 'ray', 'road', 'rocket', 'roses', 'sea', 
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 
            'snail', 'snake', 'spider', 'squirrel', 'streetcar', 
            'sunflowers', 'sweet peppers', 'table', 'tank', 'telephone', 
            'television', 'tiger', 'tractor', 'train', 'trout', 
            'tulips', 'turtle', 'wardrobe', 'whale', 'willow', 
            'wolf', 'woman', 'worm']
        print(len(self.categories))
        
    def load_model(self):
        # load json and create model
        json_file = open('cifar100/100model_70.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("cifar100/100model_70.h5")
        print("Loaded model from disk")
    
    def model_compile(self):
        epochs = 50
        lrate = 0.01
        adam = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        print(self.model.summary())
 
    def normalize(self, X_test):
        mean = 121.936
        std = 68.389
        X_test = (X_test-mean)/(std+1e-7)
        return X_test

    def model_predict(self, figure):
        test = misc.imresize(figure, (32,32,3)).transpose(2,0,1)
        test = test.astype('float32')
        test = self.normalize(test)
        test = np.expand_dims(test, axis=0)
        y_proba = self.model.predict(test)
        classed = self.model.predict_classes(test, verbose=0)
        #print(classed[0])
        if max(y_proba[0]) > 0.5:
            return(self.categories[classed[0]])
        else:
            return ''
    
    def test(self):
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
        X_test = X_test.astype('float32')
        X_test = X_test / 255.0
        # one hot encode outputs
        y_test = np_utils.to_categorical(y_test)
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))


if __name__ == '__main__':
	model = CIFAR100model()
	model.load_model()
	model.model_compile()
	model.test()

	model = VGGmodel()
	model.load_model()
	model.model_compile()
	model.test()


"""
For example:
import matplotlib.pyplot as plt
for i in range(1,9):
    filename = '../101_ObjectCategories/pizza/image_000'+str(i)+'.jpg'
    testimg = plt.imread(filename)plt.savefig('pred_airplaneside.jpg')

    print(model.model_predict(testimg))
import matplotlib.pyplot as plt
for i in range(9):
    filename = '../testfile/face'+str(i)+'.jpg'
    testimg = plt.imread(filename)
    plt.subplot(331+i)
    plt.imshow(testimg)
    plt.axis('off')
    # gray = rgb2gray(testimg)
    plt.title(model.model_predict(testimg))
plt.savefig('pred_vgg_watch.jpg')
"""