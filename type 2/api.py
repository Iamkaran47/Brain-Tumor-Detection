import os
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np

app = Flask(__name__)
model = load_model("model/braintype2.h5")
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Load and preprocess the image
        img = load_img(filepath, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_category = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_category])  # Convert to Python float

        # Get the category name
        categories = ["glioma", "meningioma", "notumor", "pituitary"]
        category_name = categories[predicted_category]

        result = {
            "category": category_name,
            "confidence": confidence,
            "image": f"/{app.config['UPLOAD_FOLDER']}/{filename}"
        }

        return jsonify(result)
    else:
        return jsonify({"error": "Invalid file format"}), 400


if __name__ == "__main__":
    app.run(debug=True)
