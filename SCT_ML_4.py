import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

# Set your LeapGestRecog dataset path here
DATASET_PATH = 'path/to/leapGestRecog'  # e.g., 'leapGestRecog'

IMG_SIZE = 128  # Images will be resized to 128x128

# Step 1: Load images & labels
images, labels = [], []
for gesture_class in sorted(os.listdir(DATASET_PATH)):
    class_dir = os.path.join(DATASET_PATH, gesture_class)
    if not os.path.isdir(class_dir):
        continue
    for img_name in os.listdir(class_dir):
        if img_name.endswith('.png'):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).convert('L').resize((IMG_SIZE, IMG_SIZE))  # Grayscale
            images.append(np.asarray(img) / 255.0)
            labels.append(gesture_class)

X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # CNN expects 4D input
le = LabelEncoder()
y = le.fit_transform(labels)
y_categorical = to_categorical(y)

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, stratify=y, random_state=42)

# Step 3: Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Step 5: Evaluate
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Test Accuracy:", accuracy_score(y_true, y_pred_classes))
print(classification_report(y_true, y_pred_classes, target_names=le.classes_))
