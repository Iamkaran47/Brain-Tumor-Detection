from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np

app = Flask(__name__)
model = load_model('model/brain.h5')
UPLOAD_FOLDER = "images"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}


# model._make_predict_function()  # Necessary for multithreading
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


IMAGE_SIZE = 128


def preprocess_image(image_path):
    image = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = 'static/uploaded_image.jpg'
            image_file.save(image_path)

            image = preprocess_image(image_path)
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction[0])
            class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
            predicted_class_name = class_names[predicted_class]
            accuracy_percentage = prediction[0][predicted_class] * 100

            return jsonify({'prediction': predicted_class_name, 'accuracy': accuracy_percentage})

    return jsonify({'error': 'Invalid request'})


if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
# from keras.models import load_model
# from keras.utils import load_img, img_to_array
# import numpy as np
# import os
# import magic  # For detecting file types
#
# app = Flask(__name__)
# model = load_model('model/brain.h5')
# UPLOAD_FOLDER = "images"
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png", "dicom"}  # Include DICOM as an allowed extension
#
# IMAGE_SIZE = 128
#
#
# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
#
#
# def is_valid_xray_image(file_path):
#     # Define allowed file types (you may need to adjust this based on your system)
#     allowed_types = ["image/jpeg", "image/png", "application/dicom"]
#
#     try:
#         # Check if the file exists
#         if not os.path.exists(file_path):
#             return False, "File not found."
#
#         # Use the 'magic' library to detect the file type
#         detected_type = magic.Magic()
#         file_type = detected_type.from_file(file_path)
#
#         # Check if the detected file type is in the allowed types
#         if file_type not in allowed_types:
#             return False, "Invalid file type."
#
#         # If the file is a DICOM image, you can perform more advanced checks
#         if file_type == "application/dicom":
#             # Add DICOM-specific checks here
#             pass
#
#         # If the file is a standard image, you can use PIL to load and check the image
#         elif file_type in ["image/jpeg", "image/png"]:
#             img = load_img(file_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#             img_array = img_to_array(img)
#
#             # Check if it's a grayscale or RGB image (customize as needed)
#             if len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] in [1, 3]):
#                 return True, "Valid X-ray image."
#
#         return False, "Invalid X-ray image format."
#
#     except Exception as e:
#         return False, str(e)
#
#
# def preprocess_image(image_path):
#     image = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#     image = img_to_array(image)
#     image = image / 255.0
#     image = np.expand_dims(image, axis=0)
#     return image
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         image_file = request.files['image']
#         if image_file and allowed_file(image_file.filename):
#             image_path = 'static/uploaded_image.jpg'
#             image_file.save(image_path)
#
#             is_valid, validation_message = is_valid_xray_image(image_path)
#             if not is_valid:
#                 return jsonify({'error': validation_message})
#
#             image = preprocess_image(image_path)
#             prediction = model.predict(image)
#             predicted_class = np.argmax(prediction[0])
#             class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
#             predicted_class_name = class_names[predicted_class]
#             accuracy_percentage = prediction[0][predicted_class] * 100
#
#             return jsonify({'prediction': predicted_class_name, 'accuracy': accuracy_percentage})
#
#         else:
#             return jsonify({'error': 'Invalid file format'})
#
#     return jsonify({'error': 'Invalid request'})
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
