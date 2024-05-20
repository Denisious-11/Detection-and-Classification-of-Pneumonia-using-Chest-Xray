

import tensorflow as tf
import tflite

# Load the pre-trained Keras model
model = tf.keras.models.load_model("Project_Saved_Models/Pneumonia_detect_92acc.h5")

# Convert the Keras model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('New_Pneumonia_detect.tflite', 'wb') as f:
    f.write(tflite_model)







##############################################################
# import tensorflow as tf

# # Load the pre-trained Keras model
# model = tf.keras.models.load_model("Project_Saved_Models/Pneumonia_detect_92acc.h5")

# # Convert the Keras model to TFLite format
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# # Save the TFLite model
# with open('Pneumonia_detect.tflite', 'wb') as f:
#     f.write(tflite_model)
