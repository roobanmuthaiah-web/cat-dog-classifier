import os
from PIL import Image

DATASET = r"D:\cat_dog_classifier\dataset"

folders = [
    os.path.join(DATASET, "train", "cats"),
    os.path.join(DATASET, "train", "dogs"),
    os.path.join(DATASET, "validation", "cats"),
    os.path.join(DATASET, "validation", "dogs")
]

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

bad_images = []


def check_images(folder):
    print(f"\nChecking: {folder}")

    for filename in os.listdir(folder):

        if not filename.lower().endswith(valid_extensions):
            continue

        path = os.path.join(folder, filename)

        try:
            with Image.open(path) as img:
                img.verify()

            # Open again because verify() invalidates the image object
            with Image.open(path) as img:
                img.convert("RGB")

        except Exception as error:
            print(f"Bad image: {path}")
            print(f"Reason: {error}")

            bad_images.append(path)


for folder in folders:
    check_images(folder)


print("\n--------------------------------")
print(f"Bad images found: {len(bad_images)}")
print("--------------------------------")


# Delete bad images
for path in bad_images:
    try:
        os.remove(path)
        print(f"Deleted: {path}")
    except Exception as error:
        print(f"Could not delete: {path}")
        print(error)


print("\nDataset cleaning completed!")