import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------------------------
# 1. Dataset paths
# --------------------------------------------------

TRAIN_DIR = r"D:\cat_dog_classifier\dataset\train"
VALIDATION_DIR = r"D:\cat_dog_classifier\dataset\validation"

# --------------------------------------------------
# 2. Settings
# --------------------------------------------------

IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32

# --------------------------------------------------
# 3. Load training images
# --------------------------------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# --------------------------------------------------
# 4. Load validation images
# --------------------------------------------------

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# --------------------------------------------------
# 5. Improve performance
# --------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# --------------------------------------------------
# 6. Create CNN model
# --------------------------------------------------

model = models.Sequential([

    # Convert pixel values from 0-255 to 0-1
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),

    # First convolution
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Second convolution
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Third convolution
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Convert feature maps into one-dimensional data
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(128, activation="relu"),

    # Output: 0 = cat, 1 = dog
    layers.Dense(1, activation="sigmoid")
])

# --------------------------------------------------
# 7. Configure training
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# 8. Display model structure
# --------------------------------------------------

model.summary()

# --------------------------------------------------
# 9. Train the model
# --------------------------------------------------

EPOCHS = 10

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# --------------------------------------------------
# 10. Save the trained model
# --------------------------------------------------

model.save("cat_dog_model.keras")

print("\nModel training completed!")
print("Model saved as cat_dog_model.keras")
