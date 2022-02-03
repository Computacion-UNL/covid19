import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import *


class InstaceModel:
    def __init__(self, ruta):
        self._ruta = ruta

    @property
    def ruta(self):
        return self._ruta

    def clasificar(file):
        imgSize = 224
        X = []
        Y = []
        #hmap = {'TEST':'test','NORMAL': 'Normal', 'COVID': 'Covid-19'}
        #label = file.split(os.path.sep)[-2]
        #print(label)
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (imgSize, imgSize))
        print(image)
        X.append(image)
        #Y.append(hmap[label])
        #print(Y.append(hmap[label]))
        #print(Y)
        return X

    def transformToTensor(X):
        le = LabelEncoder()
        #Y = le.fit_transform(Y)
        #Y = to_categorical(Y)
        testX = tf.convert_to_tensor(X, dtype=tf.uint8)
        testY = tf.Variable(X)
        testX = np.array(testX).astype('float16') / 255

        return testX, testY, X, le

    def VGG(ruta):
        try:
            # lOAD MODEL
            X = InstaceModel.clasificar(ruta)

            [testX, testY, x, le] = InstaceModel.transformToTensor(X)
            # Fin load model

            modelo = 'Models/modelVGG16to6k.h5'
            pesos_modelo = 'Models/modelVGG16to6kPesos.h5'
            cnn = load_model(modelo)
            cnn.load_weights(pesos_modelo)

            predIdxs = cnn.predict(testX)
            answer = np.argmax(predIdxs, axis=1)
            print(answer)
            if answer == 0:
                print("pred: Covid-19")
                return 0
            elif answer == 1:
                print("pred: Normal")
                return 1
            return answer
        except Exception as e:
            print(f'Ha Ocurrido un error\n {e}')
        else:
            print('No ha ocurrido ningún error')

if __name__ == "__main__":
    ruta1 = r'C:\Users\jr-98\Documents\DataCovid19imgX\TAWSIFUR_RAHMAN\COVID-19_Radiography_DatasetC\COVID\COVID-1.png'
    ruta2 = r'C:\Users\jr-98\Documents\DataCovid19imgX\TAWSIFUR_RAHMAN\COVID-19_Radiography_DatasetC\TEST\r1.png'
    InstaceModel.VGG(ruta2)