from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

import scipy.misc as misc

class ImageNetModel(object):

    def __init__(self):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.model =VGG16(weights='imagenet', include_top=True)
        self.model.summary()
    
    def model_predict(self, figure):
        test = misc.imresize(figure, (224,224,3))
        test = test.astype('float32')
        test = np.expand_dims(test, axis=0)
        test = preprocess_input(test)
        preds = self.model.predict(test)
        label = decode_predictions(preds, top=1)[0]
        return label[0][1]
    
    def test(self):
        img_path = 'elephant.jpg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        print("[INFO] classifying image...")
        preds = self.model.predict(x)
        label = decode_predictions(preds, top=1)[0]
        print(label[0][1])


if __name__ == '__main__':
    model = ImageNetModel()
    model.test()