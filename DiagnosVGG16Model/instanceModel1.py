import cv2
import tensorflow as tf
import numpy as np
from skimage import io
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import *

class InstaceModel:
    def __init__(self, url):
        self._url = url

    @property
    def ruta(self):
        return self._ruta

    def clasificar(url):
        imgSize = 224
        X = []
        image = io.imread(url)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (imgSize, imgSize))
        X.append(image)

        return X

    def transformToTensor(X):
        le = LabelEncoder()
        testX = tf.convert_to_tensor(X, dtype=tf.uint8)
        #testY = tf.Variable(X)
        testX = np.array(testX).astype('float16') / 255

        return testX, X, le

    def VGG(url):
        try:
            # lOAD MODEL
            X = InstaceModel.clasificar(url)

            [testX, x,  le] = InstaceModel.transformToTensor(X)

            # Fin load model
            modelo = 'Models/modelVGG16to6k.h5'
            pesos_modelo = 'Models/modelVGG16to6kPesos.h5'
            cnn = load_model(modelo)
            cnn.load_weights(pesos_modelo)

            predIdxs = cnn.predict(testX)
            answer = np.argmax(predIdxs, axis=1)
            print(answer)
            print(answer)
            if answer == 0:
                accuracy_score = predIdxs[0][0]
                resp = [answer, accuracy_score]
                print("pred: Covid-19")
                print(accuracy_score)
                return resp
            elif answer == 1:
                accuracy_score = predIdxs[0][1]
                resp = [answer, accuracy_score]
                print("pred: Normal")
                print(accuracy_score)
                return resp
            return answer
        except Exception as e:
            print(f'Ha Ocurrido un error\n {e}')
        else:
            print('No ha ocurrido ningún error')