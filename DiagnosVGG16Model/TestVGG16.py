import cv2
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import *
from tqdm import tqdm

# Comprobar la disponibilidad de la GPU
print(tf.test.is_gpu_available())


class TestVGG16:
    controlador = True

    def loadImage():
        imagePaths = []
        for dirname, _, filenames in os.walk(
                r'C:\Users\jr-98\Documents\DataCovid19imgX\TAWSIFUR RAHMAN\COVID-19_Radiography_Dataset'):
            for filename in filenames:
                if (filename[-3:] == 'png'):
                    imagePaths.append(os.path.join(dirname, filename))

        TestVGG16.clasificar(imagePaths)

    def clasificar(imagePaths):
        imgSize = 224
        X = []
        Y = []
        hmap = {'NORMAL': 'Normal', 'COVID': 'Covid-19'}
        for imagePath in tqdm(imagePaths):
            label = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (imgSize, imgSize))

            X.append(image)
            Y.append(hmap[label])

        # Comprobar los archivos cargados en cada una de la eqtiquetas
        print(f'Covid-19: {Y.count("Covid-19")}\n'
              f'Normal: {Y.count("Normal")}')

        le = LabelEncoder()
        Y = le.fit_transform(Y)
        Y = to_categorical(Y)

        testX = tf.convert_to_tensor(X, dtype=tf.uint8)
        testY = tf.Variable(X)

        testX = np.array(testX).astype('float16') / 255

        TestVGG16.testVgg(testY, testX, X, Y, le)

    def testVgg(testY, testX, X, Y, le):
        model = load_model('modelVGG19_6K.h5')

        try:
            predIdxs = model.predict(testX, batch_size=16)
            predIdxs = np.argmax(predIdxs, axis=1)
            print(classification_report(Y.argmax(axis=1), predIdxs, target_names=le.classes_, digits=5))
        except Exception as e:
            print(f'Ocurrio un problema{e}')
            if TestVGG16.controlador:
                print("Calculando posible caso de Covid")
                TestVGG16.intercambiarData()
                TestVGG16.loadImage()
            else:
                print(
                    'Puede que su caso no se trate de covid 19, pero se detecta ciertas anomalias que podrian agravarse'
                    'Porfavor, se le recomienda consultar su medico de cabecera')

    def intercambiarData():
        ruta1 = r'C:\Users\jr-98\Documents\DataCovid19imgX\TAWSIFUR RAHMAN\COVID-19_Radiography_Dataset\NORMAL'
        ruta2 = r'C:\Users\jr-98\Documents\DataCovid19imgX\TAWSIFUR RAHMAN\COVID-19_Radiography_Dataset\COVID'
        rutaAlt = r'C:\Users\jr-98\Documents\DataCovid19imgX\TAWSIFUR RAHMAN\COVID-19_Radiography_Dataset\COV'
        os.rename(ruta1, rutaAlt)
        os.rename(ruta2, ruta1)
        os.rename(rutaAlt, ruta2)
        TestVGG16.controlador = False


TestVGG16.loadImage()
