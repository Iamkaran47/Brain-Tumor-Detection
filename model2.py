# import tensorflow as tf
# import matplotlib.pyplot as plt
# from keras import layers
# import tensorflow_hub as hub
# import numpy as np
# import os
# import pandas as pd
# import seaborn as sns
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# train_path = "D:\Projects\TY Sem 5\Brain Tumour Detection Using CNN\Training"
# test_path = "D:\Projects\TY Sem 5\Brain Tumour Detection Using CNN\Testing"
#
# # train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)
# # test_datagen = ImageDataGenerator(rescale=1./255)
# train_ds = tf.keras.utils.image_dataset_from_directory(train_path,
#                                                      image_size=(255, 255),
#                                                      batch_size=32)
#
# test_ds = tf.keras.utils.image_dataset_from_directory(test_path,
#                                                      image_size=(255, 255),
#                                                      batch_size=32)
#
# # train_ds = train_datagen.flow_from_directory(train_path, target_size=(255, 255), batch_size=32, class_mode='categorical')
# # test_ds = test_datagen.flow_from_directory(test_path, target_size=(255, 255), batch_size=32, class_mode='categorical')
# class_names = train_ds.class_names
# print(class_names)
#
# resize_and_rescale = tf.keras.Sequential([
#     layers.Rescaling(1/255),
#     layers.Resizing(255, 255)
# ])
#
# # data_augmentation = tf.keras.Sequential([
# #     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
# #     layers.experimental.preprocessing.RandomRotation(0.2),
# # ])
#
# # AUTOTUNE = tf.data.AUTOTUNE
# # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
#
# model = tf.keras.Sequential([
#     resize_and_rescale,
#     hub.KerasLayer("https://www.kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/101-classification/versions/2", trainable=False),
#     layers.Dense(len(class_names), activation="softmax")
# ])
# model.build([None, 224, 224, 3])  # Batch input shape.
# input_shape = (255, 255, 3)
# # model.summary()
# # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # earlystop = EarlyStopping(patience=10)
# # learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
# # callbacks = [earlystop, learning_rate_reduction]
# # history = model.fit(train_ds, validation_data=test_ds, epochs=10, callbacks=callbacks)
# # model.save('model/brain.h5')
# model = tf.keras.Sequential([
#     resize_and_rescale,
#     layers.Conv2D(512, 3, activation="relu", input_shape=input_shape),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(256, 3, activation="relu"),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, 3, activation="relu"),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, 3, activation="relu"),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(32, 3, activation="relu"),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(4, activation="softmax"),
# ])
#
# model.build(input_shape=(None, 255, 255, 3))
# model.compile(loss="sparse_categorical_crossentropy",
#              optimizer=tf.keras.optimizers.Adam(),
#              metrics=["accuracy"])
# model_history = model.fit(train_ds, epochs=25, validation_data=test_ds)
# model.save('brain.h5')
#
# for image, label in test_ds.take(1):
#     plt.imshow(image[3].numpy().astype("uint8"))
#     plt.title(class_names[label[3]])
#     plt.axis("off")
#
# prediction = model.predict(tf.expand_dims(image[3], axis=0))
# prediction = np.argmax(prediction)
# prediction = class_names[prediction]

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import tensorflow as tf
from keras import layers, models
# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define paths
train_path = "D:\Projects\TY Sem 5\Brain Tumour Detection Using CNN\Training"
test_path = "D:\Projects\TY Sem 5\Brain Tumour Detection Using CNN\Testing"
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
# Add more layers as needed

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation and preprocessing
# You can use the `ImageDataGenerator` to apply data augmentation and preprocess the images before training
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create data generators for training and validation
train_generator = datagen.flow_from_directory(
    train_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_directory(
    test_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
# # Create data generators
# train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_ds = train_datagen.flow_from_directory(train_path, target_size=(255, 255), batch_size=32, class_mode='categorical')
# test_ds = test_datagen.flow_from_directory(test_path, target_size=(255, 255), batch_size=32, class_mode='categorical')
#
# # Get class names
# class_names = list(train_ds.class_indices.keys())
# print("Class Names:", class_names)
#
# # Create the model
# model = Sequential([
#     Conv2D(512, 3, activation="relu", input_shape=(255, 255, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(256, 3, activation="relu"),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, 3, activation="relu"),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, 3, activation="relu"),
#     MaxPooling2D((2, 2)),
#     Conv2D(32, 3, activation="relu"),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(len(class_names), activation="softmax"),
# ])
#
# # Compile the model
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.summary()
#
# # Define callbacks
# earlystop = EarlyStopping(patience=10)
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
# callbacks = [earlystop, learning_rate_reduction]
#
# # Train the model
# history = model.fit(train_ds, validation_data=test_ds, epochs=10, callbacks=callbacks)
#
# # Save the trained model
# model.save('model/brain.h5')
#
# # Evaluate the model on the test data
# loss, accuracy = model.evaluate(test_ds)
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)
#
# # Visualize a test image and its prediction
# for image, label in test_ds.take(1):
#     plt.imshow(image[3].numpy().astype("uint8"))
#     plt.title("True Label: " + class_names[np.argmax(label[3])])
#     plt.axis("off")
#
# prediction = model.predict(tf.expand_dims(image[3], axis=0))
# predicted_label = class_names[np.argmax(prediction)]
# print("Predicted Label:", predicted_label)
