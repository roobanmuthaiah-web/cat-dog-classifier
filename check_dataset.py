import os
import tensorflow as tf

DATASET = r"D:\cat_dog_classifier\dataset"

folders = [
    os.path.join(DATASET, "train", "cats"),
    os.path.join(DATASET, "train", "dogs"),
    os.path.join(DATASET, "validation", "cats"),
    os.path.join(DATASET, "validation", "dogs")
]

bad_images = []


for folder in folders:

    print(f"\nChecking: {folder}")

    for filename in os.listdir(folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".gif")
        ):
            continue

        path = os.path.join(folder, filename)

        try:
            image = tf.io.read_file(path)
            image = tf.image.decode_image(
                image,
                channels=3,
                expand_animations=False
            )

            # Force TensorFlow to actually process the image
            image = tf.image.resize(image, [180, 180])

        except Exception as error:

            print(f"BAD IMAGE: {path}")
            print(f"ERROR: {error}")

            bad_images.append(path)


print("\n==============================")
print(f"Bad images found: {len(bad_images)}")
print("==============================")

# Delete bad images
for path in bad_images:

    try:
        os.remove(path)
        print(f"Deleted: {path}")

    except Exception as error:
        print(f"Could not delete: {path}")
        print(error)


print("\nDataset checking completed!")