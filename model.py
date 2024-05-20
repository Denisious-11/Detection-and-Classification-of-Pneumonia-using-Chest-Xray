import numpy as np
import cv2
from PIL import Image
import os

# for train test split
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD

path_train = "D:/DENNY/Implementable_OR_not/pneumonia_chest_xray/Project_Dataset/chest_xray/train/"
classes = 2
width = 224
height = 224


def path_to_image(path_train):
    data = []
    labels = []
    for i in range(classes):
        path = path_train + '{0}/'.format(i)
        Class = os.listdir(path)
        for a in Class:
            # reading image
            image = cv2.imread(path+a)
            print(image)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            labels.append(i)
    return np.array(data), np.array(labels)


images, labels = path_to_image(path_train)

print(images.shape)
print(labels.shape)

# Data division
# Train-Test split
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Normalize the data
x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255


densenet = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)  # 32
)
model = Sequential()
model.add(densenet)
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

# compile the model
# model.compile(loss="binary_crossentropy",
#               optimizer="adam", metrics=['accuracy'])
model.compile(
    loss='binary_crossentropy',
    optimizer=SGD(
        learning_rate=0.001, momentum=0.9, name="SGD"),
    metrics=['accuracy'])

# Training
# saving the model
checkpoint = ModelCheckpoint(
    "Pneumonia_detect.h5", monitor="accuracy", save_best_only=True, verbose=1)

# training
epochs = 1
history = model.fit(x_train, y_train, batch_size=32,
                    epochs=epochs, callbacks=[checkpoint])
