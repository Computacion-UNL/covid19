import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import glob
# Use GPU witk tensorflow
import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from tensorflow.keras.callbacks import *


# Comprobar si existe o no la disponibilidad de GPU
print(tf.test.is_gpu_available())

# Directoria de las imagenes
imagePaths = []
for dirname, _, filenames in os.walk(
        r'C:\Users\jr-98\Documents\DataCovid19imgX\TAWSIFUR_RAHMAN\COVID-19_Radiography_Dataset'):
    for filename in filenames:
        if (filename[-3:] == 'png'):
            imagePaths.append(os.path.join(dirname, filename))
        else:
            print('Error con la lectura de archivos')

# Delimita la dimensiones de las imagenes
imgSize = 224

# Transformación y secmentacion de imagenes
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

# Comprobacion por lotes de las imagenes cargadas
print('Covid-19:', Y.count('Covid-19'))
print('Normal:', Y.count('Normal'))
# print('Pneumonia: ',Y.count('Pneumonia'))

le = LabelEncoder()
Y = le.fit_transform(Y)
Y = to_categorical(Y)

# Etiquetado de imagenes para al seccionde test y training de 80% traininf 20% para test
(trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.20, stratify=Y, random_state=42)

print(len(trainY))
ntimes = 6
trainY = trainY.tolist()
for i in tqdm(range(len(trainX))):
    if (trainY[i][0] == 1):
        trainX += [trainX[i]] * ntimes
        trainY += [trainY[i]] * ntimes

trainY = np.array(trainY)

print(len(trainY))

# Ajuste de las images que seran la entrada para el modelo
trainX = np.array(trainX).astype('float16')/255
testX = np.array(testX).astype('float16')/255

trainAug = ImageDataGenerator(rotation_range=20, horizontal_flip=True, fill_mode="nearest")

best_val_acc = 0
best_train_acc = 0

def saveModel(epoch, logs):
    val_acc = logs['val_accuracy']
    train_acc = logs['accuracy']
    global best_val_acc
    global best_train_acc

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save('modelVGG16to1k.h5')
        model.save_weights('modelVGG16to1Pesos.h5')
    elif val_acc == best_val_acc:
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            model.save('modelVGG16to1k.h5')
            model.save_weights('modelVGG16to1kPesos.h5')

# model
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(imgSize, imgSize, 3)))
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False

# Sumary
model.summary()

# TRAIN MODEL
INIT_LR = 3e-4
EPOCHS = 50
BS = 32

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    callbacks=[LambdaCallback(on_epoch_end=saveModel), ],
    epochs=EPOCHS)

# Plot de las resultados de entrenamiento
N = EPOCHS
plt.style.use("ggplot")

plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch Number")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.figure(figsize=(20, 20))

from sklearn.metrics import accuracy_score

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
t = f.suptitle('Transfer Learning VGG16 Performance', fontsize=16, fontweight='bold')
f.subplots_adjust(top=0.9, wspace=0.1)

max_epoch = len(H.history['accuracy']) + 1
epoch_list = list(range(1, max_epoch))
ax1.plot(epoch_list, H.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, H.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 1))
ax1.set_ylabel('Accuracy Value', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax1.set_title('Accuracy', fontsize=14, fontweight='bold')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, H.history['loss'], label='Train Loss')
ax2.plot(epoch_list, H.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(1, max_epoch, 1))
ax2.set_ylabel('Loss Value', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax2.set_title('Loss', fontsize=14, fontweight='bold')
l2 = ax2.legend(loc="best")

# LOAD THE BEST MODEL
model = load_model('modelVGG16to1k.h5')

# RESULT OF TRAIN

predIdxs = model.predict(trainX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(trainY.argmax(axis=1), predIdxs, target_names=le.classes_, digits=5))

# Result TEST

predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=le.classes_, digits=5))

# Graficas de prediccion
import seaborn as sns
import sklearn

plt.figure()

ax = plt.subplot()

ax.set_title('Confusion Matrix')
pred = model.predict(testX)
pred = np.argmax(pred, axis=1)
# pred = model.predict_classes(X_test)
Y_TEST = np.argmax(testY, axis=1)
cm = sklearn.metrics.confusion_matrix(Y_TEST, pred)
classes = ['normal', 'covid19']
sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes, cmap='Blues')

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show

