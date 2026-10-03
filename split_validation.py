import os
import shutil
import random

DATASET = r"D:\cat_dog_classifier\dataset"

TRAIN_RATIO = 0.8


def create_validation(category):
    train_folder = os.path.join(DATASET, "train", category)
    validation_folder = os.path.join(DATASET, "validation", category)

    # Create validation folder if it doesn't exist
    os.makedirs(validation_folder, exist_ok=True)

    # Get images from training folder
    images = [
        file for file in os.listdir(train_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Randomize
    random.shuffle(images)

    # Number of images that should remain in training
    train_count = int(len(images) * TRAIN_RATIO)

    # Images after the first 80%
    validation_images = images[train_count:]

    print(f"\n{category.upper()}")
    print(f"Total images: {len(images)}")
    print(f"Moving to validation: {len(validation_images)}")

    # Move 20% to validation
    for image in validation_images:
        source = os.path.join(train_folder, image)
        destination = os.path.join(validation_folder, image)

        shutil.move(source, destination)


create_validation("cats")
create_validation("dogs")

print("\nDataset split completed!")