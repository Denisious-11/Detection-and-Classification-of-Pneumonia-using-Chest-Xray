import numpy as np
import cv2
from PIL import Image
import os

# for train test split
from sklearn.model_selection import train_test_split


path_train = "D:/DENNY/Implementable_OR_not/pneumonia_chest_xray/chest_xray/train/"
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
print(labels)
# print(images)
print(images.shape)
# print(labels)
print(labels.shape)

# Data division
# Train-Test split
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
