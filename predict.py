import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("cat_dog_model.keras")


IMAGE_PATH = r"D:\cat_dog_classifier\test.jpg\4.jpg"

# Load and resize image
image = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=(180, 180)
)


image_array = tf.keras.utils.img_to_array(image)


image_array = np.expand_dims(image_array, axis=0)


prediction = model.predict(image_array, verbose=0)[0][0]


if prediction < 0.5:
    print("Prediction: CAT 🐱")
    print(f"Confidence: {(1 - prediction) * 100:.2f}%")
else:
    print("Prediction: DOG 🐶")
    print(f"Confidence: {prediction * 100:.2f}%")
