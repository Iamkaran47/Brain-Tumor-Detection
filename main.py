# # Libraries Data Visualization
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Libraries for Data Pre-Processing
# import numpy as np
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from PIL import Image, ImageEnhance
#
# # Libraries for Model Creation
# from tensorflow import keras
# from keras.layers import *
# from keras.losses import *
# from keras.models import *
# from keras.metrics import *
# from keras.optimizers import *
# from keras.applications import *
# from keras.utils import load_img, img_to_array
# from keras.models import save_model
#
# # General Purpose Libraries
# from tqdm import tqdm
# import os
# import random
#
# print('Libraries imported successfully!')
# train_dir = "D:\Projects\TY Sem 5\Brain Tumour Detection Using CNN\Training"
# test_dir = "D:\Projects\TY Sem 5\Brain Tumour Detection Using CNN\Testing"
#
# train_paths = []
# train_labels = []
#
# for label in os.listdir(train_dir):
#     for image in os.listdir(train_dir + '//' + label):
#         train_paths.append(train_dir + '//' + label + '//' + image)
#         train_labels.append(label)
#
# train_paths, train_labels = shuffle(train_paths, train_labels)
# print('Training set setup complete!')
#
# validation_ratio = 0.1
#
# test_paths = []
# test_labels = []
# validation_paths = []
# validation_labels = []
#
# for label in os.listdir(test_dir):
#     label_dir = test_dir + '//' + label
#     images = os.listdir(label_dir)
#     random.shuffle(images)
#
#     num_validation = int(len(images) * validation_ratio)
#
#     validation_images = images[:num_validation]
#     testing_images = images[num_validation:]
#
#     validation_paths.extend([label_dir + '//' + img for img in validation_images])
#     validation_labels.extend([label] * len(validation_images))
#     test_paths.extend([label_dir + '//' + img for img in testing_images])
#     test_labels.extend([label] * len(testing_images))
#
# validation_data = list(zip(validation_paths, validation_labels))
# random.shuffle(validation_data)
# validation_paths, validation_labels = zip(*validation_data)
#
# test_data = list(zip(test_paths, test_labels))
# random.shuffle(test_data)
# test_paths, test_labels = zip(*test_data)
#
# def augment_image(image):
#     image = Image.fromarray(np.uint8(image))
#     image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.3))
#     image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.3))
#     image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.9, 1.3))
#
#     # Additional Data Augmentation
#     if random.random() > 0.5:
#         image = image.transpose(Image.FLIP_LEFT_RIGHT)
#
#     angle = random.uniform(-10, 10)
#     image = image.rotate(angle)
#
#     image = np.array(image) / 255.0
#     return image
#
#
# print('Image enhancement setup complete!')
#
#
# IMAGE_SIZE = 128
#
#
# def open_images(paths):
#     images = []
#
#     for path in paths:
#         image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#         image = augment_image(image)
#         images.append(image)
#
#     return np.array(images)
#
#
# images = open_images(train_paths[50:59])
# labels = train_labels[50:59]
# fig = plt.figure(figsize=(12, 6))
#
# for x in range(1, 9):
#     fig.add_subplot(2, 4, x)
#     plt.axis('off')
#     plt.title(labels[x])
#     plt.imshow(images[x])
#
# plt.rcParams.update({'font.size': 12})
# plt.show()
#
#
# unique_labels = os.listdir(train_dir)
#
#
# def encode_label(labels):
#     encoded = []
#
#     for x in labels:
#         encoded.append(unique_labels.index(x))
#
#     return np.array(encoded)
#
#
# def decode_label(labels):
#     decoded = []
#
#     for x in labels:
#         decoded.append(unique_labels[x])
#
#     return np.array(decoded)
#
#
# def datagen(paths, labels, batch_size=12, epochs=1):
#     for _ in range(epochs):
#         for x in range(0, len(paths), batch_size):
#             batch_paths = paths[x: x + batch_size]
#             batch_images = open_images(batch_paths)
#             batch_labels = labels[x: x + batch_size]
#             batch_labels = encode_label(batch_labels)
#             yield batch_images, batch_labels
#
#
# # def build_model():
# #     model = Sequential()
# #     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
# #     model.add(MaxPooling2D((2, 2)))
# #     model.add(Conv2D(64, (3, 3), activation='relu'))
# #     model.add(MaxPooling2D((2, 2)))
# #     model.add(Conv2D(128, (3, 3), activation='relu'))
# #     model.add(MaxPooling2D((2, 2)))
# #     model.add(Conv2D(64, (3, 3), activation='relu'))
# #     model.add(MaxPooling2D((2, 2)))
# #     model.add(Conv2D(32, (3, 3), activation='relu'))
# #     model.add(Flatten())
# #     model.add(Dense(16, activation='relu'))
# #     model.add(Dense(4, activation='softmax'))
# #     model.compile(optimizer='adam',
# #                   loss='sparse_categorical_crossentropy',
# #                   metrics=['accuracy'])
# #     return model
# #
# #
# # model = build_model()
# # model.summary()
# def build_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#
#     # Reduce Model Complexity
#     model.add(Dense(64, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dense(32, activation='relu'))
#     model.add(BatchNormalization())
#
#     model.add(Dense(4, activation='softmax'))
#
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model
#
#
# model = build_model()
# model.summary()
#
# model_info = model.fit(
#     datagen(
#         train_paths,
#         train_labels,
#         batch_size=32,
#         epochs=20
#     ),
#     steps_per_epoch=len(train_paths) // 32,
#     validation_data=datagen(
#         validation_paths,
#         validation_labels,
#         batch_size=12,
#         epochs=20
#     ),
#     validation_steps=len(validation_paths) // 12,
#     epochs=20)
# print('Training complete!')
#
# prediction = model.predict(open_images(test_paths))
# prediction = np.argmax(prediction, axis=1)
# print(classification_report(encode_label(test_labels), prediction, target_names=unique_labels))
#
# save_model(model, 'D:\Projects\TY Sem 5\Brain Tumour Detection Using CNN/brain.h5')

# Libraries Data Visualization


from sklearn.utils import shuffle, compute_class_weight

from PIL import Image, ImageEnhance
import os
import random
import numpy as np
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import load_img, img_to_array
from keras.models import save_model
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import *
import matplotlib.pyplot as plt
from keras.utils import *
from tqdm import keras

print('Libraries imported successfully!')

train_dir = "D:\Projects\TY Sem 5\Brain Tumour Detection Using CNN\Training"
test_dir = "D:\Projects\TY Sem 5\Brain Tumour Detection Using CNN\Testing"

train_paths = []
train_labels = []

for label in os.listdir(train_dir):
    for image in os.listdir(os.path.join(train_dir, label)):
        train_paths.append(os.path.join(train_dir, label, image))
        train_labels.append(label)

train_paths, train_labels = shuffle(train_paths, train_labels)
print('Training set setup complete!')

validation_ratio = 0.1

test_paths = []
test_labels = []
validation_paths = []
validation_labels = []

for label in os.listdir(test_dir):
    label_dir = os.path.join(test_dir, label)
    images = os.listdir(label_dir)
    random.shuffle(images)

    num_validation = int(len(images) * validation_ratio)

    validation_images = images[:num_validation]
    testing_images = images[num_validation:]

    validation_paths.extend([os.path.join(label_dir, img) for img in validation_images])
    validation_labels.extend([label] * len(validation_images))
    test_paths.extend([os.path.join(label_dir, img) for img in testing_images])
    test_labels.extend([label] * len(testing_images))

validation_data = list(zip(validation_paths, validation_labels))
random.shuffle(validation_data)
validation_paths, validation_labels = zip(*validation_data)


def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.3))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.3))
    image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.9, 1.3))

    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    angle = random.uniform(-10, 10)
    image = image.rotate(angle)

    image = np.array(image) / 255.0
    return image


print('Image enhancement setup complete!')

# Add the following line to define the IMAGE_SIZE
IMAGE_SIZE = 128


# def open_images(paths):
#     images = []
#
#     for path in paths:
#         image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#         image = augment_image(image)
#         images.append(image)
#
#     return np.array(images)
def open_images(paths, target_size=(128, 128)):
    images = []
    for path in paths:
        image = load_img(path, target_size=target_size)
        image = img_to_array(image) / 255.0
        images.append(image)
    images = np.array(images)
    return images


unique_labels = os.listdir(train_dir)


def encode_label(labels):
    encoded = [unique_labels.index(label) for label in labels]
    return np.array(encoded)


def decode_label(labels):
    decoded = [unique_labels[label_idx] for label_idx in labels]
    return np.array(decoded)


# Calculate class weights for the entire dataset
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = dict(enumerate(class_weights))


def is_brain_mri(image_path):
    # Load the image
    image = Image.open(image_path)

    # Check metadata or visual features to determine if it's an MRI of the brain
    # You can implement your own criteria here based on the expected properties of brain MRIs

    return True  # Return True if the image is determined to be a brain MRI, otherwise False


def datagen(paths, labels, class_weights, target_size=(128, 128), batch_size=12, epochs=1):
    num_samples = len(paths)
    for _ in range(epochs):
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_paths = paths[start:end]

            # Filter out non-brain MRI images
            batch_paths = [img_path for img_path in batch_paths if is_brain_mri(img_path)]

            batch_images = open_images(batch_paths, target_size=target_size)
            batch_labels = labels[start:end]

            # Convert batch_labels to encoded integer labels
            batch_labels_encoded = encode_label(batch_labels)

            yield batch_images, batch_labels_encoded


# def datagen(paths, labels, class_weights, target_size=(128, 128), batch_size=12, epochs=1):
#     num_samples = len(paths)
#     for _ in range(epochs):
#         for start in range(0, num_samples, batch_size):
#             end = min(start + batch_size, num_samples)
#             batch_paths = paths[start:end]
#             batch_images = [load_img(path, target_size=target_size) for path in batch_paths]
#             batch_images = [img_to_array(img) for img in batch_images]
#             batch_images = np.array(batch_images) / 255.0
#             batch_labels = labels[start:end]
#
#             # Convert batch_labels to encoded integer labels
#             batch_labels_encoded = encode_label(batch_labels)
#
#             yield batch_images, batch_labels_encoded    real


# def datagen(paths, labels, batch_size=12, epochs=1, apply_class_weights=True):
#     num_samples = len(paths)
#     for _ in range(epochs):
#         for start in range(0, num_samples, batch_size):
#             end = min(start + batch_size, num_samples)
#             batch_paths = paths[start:end]
#             batch_images = open_images(batch_paths)
#             batch_labels = labels[start:end]
#             batch_labels_encoded = encode_label(batch_labels)
#
#             # Calculate class weights for each batch if needed
#             if apply_class_weights:
#                 class_weights = compute_class_weight('balanced', classes=np.unique(batch_labels_encoded),
#                                                      y=batch_labels_encoded)
#                 class_weights_dict = dict(enumerate(class_weights))
#             else:
#                 class_weights_dict = None
#
#             yield batch_images, batch_labels_encoded, class_weights_dict


# def build_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#
#     # Reduce Model Complexity
#     model.add(Dense(64, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dense(32, activation='relu'))
#     model.add(BatchNormalization())
#
#     model.add(Dense(4, activation='softmax'))
#
#     class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
#     class_weights_dict = dict(enumerate(class_weights))
#
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#                   # class_weight=class_weights_dict)  # Pass the class weights directly to the compile() function
#     return model

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    # Reduce Model Complexity
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model.compile(optimizer='rmsprop',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])

    return model


model = build_model()
model.summary()


# Learning Rate Scheduling
def lr_schedule(epoch):
    initial_learning_rate = 0.001
    decay_rate = 0.5
    # decay_rate = 0.1
    # decay_steps = 5
    decay_steps = 2
    lr = initial_learning_rate * (decay_rate ** np.floor(epoch / decay_steps))
    return lr


lr_scheduler = LearningRateScheduler(lr_schedule)

# Early Stopping
early_stopping = EarlyStopping(patience=8, restore_best_weights=True)
# from keras import regularizers
# model.add(Dense(64, input_dim=64,
#                 kernel_regularizer=regularizers.l2(0.01),

model_info = model.fit(
    datagen(
        train_paths,
        train_labels,
        class_weights=class_weights_dict,
        target_size=(128, 128),
        batch_size=32,
        epochs=20
    ),
    steps_per_epoch=len(train_paths) // 32,
    validation_data=datagen(
        validation_paths,
        validation_labels,
        class_weights=None,
        target_size=(128, 128),
        batch_size=12,
        epochs=20
    ),
    validation_steps=len(validation_paths) // 12,
    epochs=20,
    callbacks=[lr_scheduler, early_stopping]
)


# model_info = model.fit(
#     datagen(
#         train_paths,
#         train_labels,
#         class_weights=class_weights_dict,
#         target_size=(128, 128),
#         batch_size=32,
#         epochs=20
#     ),
#     steps_per_epoch=len(train_paths) // 32,
#     validation_data=datagen(
#         validation_paths,
#         validation_labels,
#         class_weights=None,
#         target_size=(128, 128),
#         batch_size=12,
#         epochs=20
#     ),
#     validation_steps=len(validation_paths) // 12,
#     epochs=20,
#     callbacks=[lr_scheduler, early_stopping]
# real )

print('Training complete!')
plt.figure(figsize = (20, 15))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(model_info.history['accuracy'], color = 'b', label = "Training Accuracy")
plt.plot(model_info.history['val_accuracy'], color = 'r', label = "Validation Accuracy")
plt.legend()
plt.grid()
plt.show()

prediction = model.predict(open_images(test_paths))
prediction = np.argmax(prediction, axis=1)
print(classification_report(encode_label(test_labels), prediction, target_names=unique_labels))

save_model(model, 'D:\Projects\TY Sem 5\Brain Tumour Detection Using CNN/brain.h5')
