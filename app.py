from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("cat_dog_model.keras")


@app.route("/")
def home():
    return "Cat Dog Classifier API is running!"


@app.route("/predict", methods=["POST"])
def predict():

    # Check whether an image was uploaded
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    image_file = request.files["image"]

    # Convert uploaded image into 180x180
    image = tf.keras.utils.load_img(
        image_file,
        target_size=(180, 180)
    )

    # Convert image to numbers
    image_array = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Ask the model for prediction
    prediction = model.predict(image_array, verbose=0)[0][0]

    # Convert prediction to result
    if prediction < 0.5:
        label = "cat"
        confidence = (1 - prediction) * 100
    else:
        label = "dog"
        confidence = prediction * 100

    return jsonify({
        "prediction": label,
        "confidence": round(float(confidence), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)