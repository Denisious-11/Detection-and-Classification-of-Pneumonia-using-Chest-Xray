import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

loaded_model = load_model(
    "Project_Saved_Models/Pneumonia_cause_detect_95acc.h5")

path_test = "D:/DENNY/Implementable_OR_not/pneumonia_chest_xray/Test/Test2/"
width = 224
height = 224
data = []
image = cv2.imread(path_test + "/b9.jpeg")
image_from_array = Image.fromarray(image, 'RGB')
size_image = image_from_array.resize((height, width))
data.append(np.array(size_image))
x_test = np.array(data)
# print(">>>>>>>>>>>>",x_test)

x_test = x_test/255

my_pred = loaded_model.predict(x_test)
print(my_pred)
my_pred = my_pred[0]
print(my_pred)
my_pred = my_pred[0]
print(my_pred)

if my_pred > 0.8:
    print("RESULT: Pneumonia Detected : VIRUS")
elif my_pred <= 0.8:
    print("RESULT: Pneumonia Detected : BACTERIA")
