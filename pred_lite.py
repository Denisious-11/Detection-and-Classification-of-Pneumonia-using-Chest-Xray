import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter(model_path="New_Pneumonia_detect.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print("*********")

# Prepare a single image for prediction
image_path = "person1_bacteria_1.jpeg"
input_shape = input_details[0]['shape']
input_image = cv2.imread(image_path)
input_image = cv2.resize(input_image, (224, 224))
input_image = input_image / 255.0  # Normalize the input
input_image = np.float32(input_image)  # Convert input to float32
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

# Perform inference
interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Interpret the output (assuming binary classification)
predicted_class = "Pneumonia" if output_data[0][0] > 0.5 else "Normal"
print("Predicted class:", predicted_class)



#######################
# import numpy as np
# import cv2
# from PIL import Image
# # import tensorflow as tf

# # Load the TFLite model
# interpreter = tf.lite.Interpreter(model_path="Pneumonia_detect.tflite")
# interpreter.allocate_tensors()


# # Get input and output details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# print(input_details)
# print("*********")

# # Prepare a single image for prediction
# image_path = "person1_bacteria_1.jpeg"
# input_shape = input_details[0]['shape']
# input_image = cv2.imread(image_path)
# input_image = cv2.resize(input_image, (224, 224))
# input_image = input_image / 255.0  # Normalize the input
# input_image = np.float32(input_image)  # Convert input to float32
# input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

# # Perform inference
# interpreter.set_tensor(input_details[0]['index'], input_image)
# interpreter.invoke()

# # Get the output tensor
# output_data = interpreter.get_tensor(output_details[0]['index'])

# # Interpret the output (assuming binary classification)
# predicted_class = "Pneumonia" if output_data[0][0] > 0.5 else "Normal"
# print("Predicted class:", predicted_class)
