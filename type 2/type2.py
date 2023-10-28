import os
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.saving.saving_api import save_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import layers
from sklearn.metrics import classification_report
from keras.utils import plot_model
from tensorflow import keras
import warnings
from IPython.display import HTML, display

warnings.filterwarnings("ignore")


def read_images(folder_path, image_size=(256, 256)):
    image_list = []
    label_list = []

    for root, subdirs, files in os.walk(folder_path):
        for subdir in subdirs:
            label = subdir

            subdir_path = os.path.join(root, subdir)
            for file in os.listdir(subdir_path):
                image_path = os.path.join(subdir_path, file)

                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, image_size)

                image_list.append(image)
                label_list.append(label)
    images = np.array(image_list)
    labels = np.array(label_list)

    return images, labels


testing_images, testing_labels = read_images("D:\Projects\TY Sem 5\Brain Tumour Detection Using CNN\Testing")
training_images, training_labels = read_images("D:\Projects\TY Sem 5\Brain Tumour Detection Using CNN\Training")

print(testing_images.shape, testing_labels.shape)
print(training_images.shape, training_labels.shape)

train_size = len(training_images)
test_size = len(testing_images)

sizes = [train_size, test_size]
labels = ['Training', 'Testing']
colors = ['skyblue', 'lightcoral']
explode = (0.1, 0)

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', shadow=True)
plt.title("Proportion of Training and Testing Datasets")
plt.axis('equal')
plt.savefig('pie-00.png')
plt.close()
with open("pie-00.png", "rb") as img_file:
    img_data = img_file.read()

import base64

img_base64 = base64.b64encode(img_data).decode("utf-8")

num_images_to_display = 4
random_indices = np.random.choice(len(training_images), size=num_images_to_display, replace=False)

fig, axes = plt.subplots(1, num_images_to_display, figsize=(10, 4))

for i, index in enumerate(random_indices):
    image = training_images[index]
    label = training_labels[index]
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('output_images.png')
plt.close()
with open("output_images.png", "rb") as img_file:
    img_data = img_file.read()

import base64

img_base64 = base64.b64encode(img_data).decode("utf-8")

num_images_to_display = 4
random_indices = np.random.choice(len(testing_images), size=num_images_to_display, replace=False)

fig, axes = plt.subplots(1, num_images_to_display, figsize=(10, 4))

for i, index in enumerate(random_indices):
    image = testing_images[index]
    label = testing_labels[index]
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('output_images.png')
plt.close()
with open("output_images.png", "rb") as img_file:
    img_data = img_file.read()

import base64

img_base64 = base64.b64encode(img_data).decode("utf-8")

unique_labels, label_counts = np.unique(training_labels, return_counts=True)

plt.figure(figsize=(6, 6))
explode = [0.1] * len(unique_labels)
colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
shadow = True

plt.pie(label_counts, labels=unique_labels, explode=explode, autopct='%1.1f%%', colors=colors, shadow=shadow)
plt.title("Class Distribution in Train")
plt.savefig('output_pie_chart.png')
plt.close()

with open("output_pie_chart.png", "rb") as img_file:
    img_data = img_file.read()

import base64

img_base64 = base64.b64encode(img_data).decode("utf-8")

testing_indices = np.random.permutation(testing_images.shape[0])
testing_images = testing_images[testing_indices] / 255.0
testing_labels = testing_labels[testing_indices]

training_indices = np.random.permutation(training_images.shape[0])
training_images = training_images[training_indices] / 255.0
training_labels = training_labels[training_indices]

print(testing_images.shape, testing_labels.shape)
print(training_images.shape, training_labels.shape)

print("----------------------")
print(training_images[0])
print(training_labels[0])
print("----------------------")
print(testing_images[0])
print(testing_labels[0])
print("----------------------")

train_images = np.squeeze(training_images)
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.squeeze(testing_images)
test_images = np.expand_dims(test_images, axis=-1)
print(train_images.shape)
print(test_images.shape)

train_labels = np.squeeze(training_labels)
test_labels = np.squeeze(testing_labels)
print(train_labels.shape, test_labels.shape)

print(train_images.shape, test_images.shape)

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

num_classes = len(label_encoder.classes_)
train_labels_onehot = to_categorical(train_labels_encoded, num_classes=num_classes)
test_labels_onehot = to_categorical(test_labels_encoded, num_classes=num_classes)

print(train_labels_encoded.shape)
print(test_labels_encoded.shape)

print(train_labels_onehot.shape)
print(test_labels_onehot.shape)

from keras.optimizers import Adam

model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
num_labels = 4
model.add(layers.Dense(num_labels, activation='softmax'))
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.0,
    horizontal_flip=True
)

datagen.fit(train_images)
train_generator = datagen.flow(train_images, train_labels_onehot, batch_size=32)
test_generator = datagen.flow(test_images, test_labels_onehot, batch_size=32)

print("Train Generator Shape:", train_generator[0][0].shape)
print("Test  Generator Shape:", test_generator[0][0].shape)

model.fit(train_generator, epochs=10, validation_data=test_generator)

predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
target_names = label_encoder.classes_
print(classification_report(test_labels_encoded, predicted_labels, target_names=target_names))

save_model(model, 'D:\Projects\TY Sem 5\Brain Tumour Detection Using CNN\\type 2\\brain2.h5')