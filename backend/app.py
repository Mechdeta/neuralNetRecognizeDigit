from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io
import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
MODEL_PATH = os.path.join(BASE_DIR, "mnist_model.keras")

# ---------- App (MUST COME FIRST) ----------
app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=""
)

# ---------- Load model ----------
model = tf.keras.models.load_model(MODEL_PATH)

# ---------- Routes ----------
@app.route("/")
def home():
    return "MNIST Digit Recognition API is running"

@app.route("/ui")
def ui():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]

        # Open image
        image = Image.open(io.BytesIO(file.read())).convert("L")

        # Invert colors
        image = ImageOps.invert(image)

        # Crop digit
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)

        # Resize correctly
        image = image.resize((28, 28), Image.LANCZOS)

        # Convert to numpy
        img = np.array(image) / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img, verbose=0)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            "predicted_digit": digit,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
