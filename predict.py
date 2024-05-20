import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

loaded_model = load_model("Project_Saved_Models/Pneumonia_detect_92acc.h5")

path_test = "Test/Test1/"
width = 224
height = 224
data = []
image = cv2.imread(path_test + "/yes_3.jpeg")
image_from_array = Image.fromarray(image, 'RGB')
size_image = image_from_array.resize((height, width))
data.append(np.array(size_image))
x_test = np.array(data)
# print(">>>>>>>>>>>>",x_test)

x_test=x_test/255

my_pred = loaded_model.predict(x_test)
print(my_pred)
my_pred = my_pred[0]
print(my_pred)
my_pred = my_pred[0]
print(my_pred)

if my_pred > 0.5:
    print("RESULT: Pneumonia Detected")
elif my_pred <= 0.5:
    print("RESULT: Pneumonia NOT Detected")
