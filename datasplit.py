import os
import shutil
from sklearn.model_selection import train_test_split


def create_train_test_split(source_dir, train_dir, test_dir, test_size=0.2):
    # Ensure the destination directories exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Iterate over each folder in the source directory
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if os.path.isdir(category_path):
            images = os.listdir(category_path)
            train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)

            # Create category directories in train and test directories
            train_category_dir = os.path.join(train_dir, category)
            test_category_dir = os.path.join(test_dir, category)
            os.makedirs(train_category_dir, exist_ok=True)
            os.makedirs(test_category_dir, exist_ok=True)

            # Copy training images
            for image in train_images:
                shutil.copy(os.path.join(category_path, image), os.path.join(train_category_dir, image))

            # Copy testing images
            for image in test_images:
                shutil.copy(os.path.join(category_path, image), os.path.join(test_category_dir, image))

    print("Dataset split completed successfully!")


source_dir = './dataset'  # Replace with the path to your dataset
train_dir = './Data-Set/train'
test_dir = './Data-Set/test'
create_train_test_split(source_dir, train_dir, test_dir)
