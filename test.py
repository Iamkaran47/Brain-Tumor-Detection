# import tensorflow as tf
# from keras.preprocessing.image import *
# from keras.utils import load_img, img_to_array
# import numpy as np
#
# # Load the trained model
# model = tf.keras.models.load_model('brain.h5')
#
# # Define the class labels
# class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
#
# # Provide the path to your test image
# test_image_path = 'D:\Brain_tumor_dataset\\No\\No15.jpg'
#
# # Set the image dimensions based on your model's input requirements
# image_width = 128
# image_height = 128
#
# # Load and preprocess the input image
# test_image = load_img(test_image_path, target_size=(image_width, image_height))
# test_image_array = img_to_array(test_image)
# test_image_array = np.expand_dims(test_image_array, axis=0)
# preprocessed_data = test_image_array / 255.0  # Normalize pixel values to [0, 1]
#
# # Perform prediction
# predictions = model.predict(preprocessed_data)
#
# # Get the predicted class index and accuracy
# predicted_class_index = np.argmax(predictions)
# accuracy_percentage = predictions[0][predicted_class_index] * 100
#
# # Get the predicted class label
# predicted_class_label = class_labels[predicted_class_index]
#
# # Print the results
# print("Predicted Class:", predicted_class_label)
# print("Accuracy Percentage:", accuracy_percentage)

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import *
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array

# Load the trained model
model = tf.keras.models.load_model('braintype2.h5')

# Define the class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Provide the path to your test image or directory
test_path = "C:\\Users\\theno\OneDrive\Desktop\PituitaryAdenoma.jpg" #Replace with your test image path or directory

# Set the image dimensions based on your model's input requirements
image_width = 128
image_height = 128

# Check if the provided path is a directory or a single image
if os.path.isdir(test_path):
    # If the path is a directory
    test_datagen = ImageDataGenerator(rescale=1. / 255)  # Normalize pixel values to [0, 1]
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(image_width, image_height),
        batch_size=12,
        class_mode='categorical',
        shuffle=False
    )

    # Perform prediction for each image in the directory
    predictions = model.predict(test_generator)

    # Iterate over the predictions
    for i in range(len(test_generator.filenames)):
        # Get the predicted class index and accuracy for the current image
        predicted_class_index = np.argmax(predictions[i])
        accuracy_percentage = predictions[i][predicted_class_index] * 100

        # Get the predicted class label
        predicted_class_label = class_labels[predicted_class_index]

        # Print the results for the current image
        print("Image:", test_generator.filenames[i])
        print("Predicted Class:", predicted_class_label)
        print("Accuracy Percentage:", accuracy_percentage)
        print("----------------------")

else:
    # If the path is a single image
    test_image = load_img(test_path, target_size=(image_width, image_height))
    test_image_array = img_to_array(test_image)
    test_image_array = np.expand_dims(test_image_array, axis=0)
    preprocessed_data = test_image_array / 255.0  # Normalize pixel values to [0, 1]

    # Perform prediction
    predictions = model.predict(preprocessed_data)

    # Get the predicted class index and accuracy
    predicted_class_index = np.argmax(predictions)
    accuracy_percentage = predictions[0][predicted_class_index] * 100

    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]

    # Print the results for the single image
    print("Image:", test_path)
    print("Predicted Class:", predicted_class_label)
    print("Accuracy Percentage:", accuracy_percentage)
