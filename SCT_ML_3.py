import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from PIL import Image

# Set these paths to the downloaded Kaggle dataset locations
cat_dir = "path/to/train/cat"  # e.g., "./train/cat"
dog_dir = "path/to/train/dog"  # e.g., "./train/dog"

# Parameters
IMG_SIZE = 64  # Resize all images to 64x64 pixels

def load_images_from_folder(folder, label, img_size):
    images, labels = [], []
    for filename in tqdm(os.listdir(folder)):
        if filename.endswith(".jpg"):
            try:
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
                img_np = np.asarray(img) / 255.0  # Normalize to 0-1
                images.append(img_np.flatten())  # Flatten 3D image to 1D array
                labels.append(label)
            except:
                continue
    return images, labels

# Load data
cat_images, cat_labels = load_images_from_folder(cat_dir, 0, IMG_SIZE)  # Label 0 for cats
dog_images, dog_labels = load_images_from_folder(dog_dir, 1, IMG_SIZE)  # Label 1 for dogs

X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM
print("Training SVM. This may take several minutes for large datasets...")
clf = SVC(kernel='linear', verbose=True)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
