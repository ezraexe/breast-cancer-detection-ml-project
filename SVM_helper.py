import os
import cv2
import numpy as np
from tqdm import tqdm

def load_image_data(path):
    """
    Args:
        path (str): Path to test or training set
    """
    images = []
    labels = []

    benign_path = os.path.join(path, 'benign')
    malignant_path = os.path.join(path, 'malignant')
    total_images = len(os.listdir(benign_path)) + len(os.listdir(malignant_path))
    progress = tqdm(total=total_images, desc="Processing images", mininterval=1)

    # Process benign images
    for image_name in os.listdir(benign_path):
        image_path = os.path.join(benign_path, image_name)
        img = cv2.imread(image_path)
        resize_img = cv2.resize(img, (128, 128))
        images.append(resize_img.flatten())
        labels.append(0)
        progress.update(1)

    # Process malignant images
    for image_name in os.listdir(malignant_path):
        image_path = os.path.join(malignant_path, image_name)
        img = cv2.imread(image_path)
        resize_img = cv2.resize(img, (128, 128))
        images.append(resize_img.flatten())
        labels.append(1)
        progress.update(1)
    
    progress.close()
    return np.array(images), np.array(labels)
