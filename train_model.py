import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load dataset
data_dir = r"D:\EMOTION REC\dataset"
img_size = 48
data = []
labels = []

for emotion in os.listdir(data_dir):
    folder = os.path.join(data_dir, emotion)
    if not os.path.isdir(folder):
        continue
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        data.append(img)
        labels.append(emotion)

X = np.array(data).reshape(-1, img_size, img_size, 1) / 255.0
le = LabelEncoder()
y = le.fit_transform(labels)
y = to_categorical(y)

# Save label encoder
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Augment + Train
aug = ImageDataGenerator(rotation_range=15, zoom_range=0.2, horizontal_flip=True)
model.fit(aug.flow(X_train, y_train, batch_size=32), epochs=25, validation_data=(X_test, y_test))

model.save("emotion_model.h5")
print("Model trained and saved as emotion_model.h5")
