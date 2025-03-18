import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import BatchNormalization

X = []
Z = []
IMG_SIZE = 150

FLOWER_DAISY_DIR = './flowers/daisy'
FLOWER_SUNFLOWER_DIR = './flowers/sunflower'
FLOWER_TULIP_DIR = './flowers/tulip'
FLOWER_DANDI_DIR = './flowers/dandelion'
FLOWER_ROSE_DIR = './flowers/rose'

def assign_label(img, flower_type):
    return flower_type

def make_train_data(flower_type, DIR):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img, flower_type)
        path = os.path.join(DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(np.array(img))
        Z.append(str(label))

# Load Data
make_train_data('Daisy', FLOWER_DAISY_DIR)
make_train_data('Sunflower', FLOWER_SUNFLOWER_DIR)
make_train_data('Tulip', FLOWER_TULIP_DIR)
make_train_data('Dandelion', FLOWER_DANDI_DIR)
make_train_data('Rose', FLOWER_ROSE_DIR)

# Convert to NumPy arrays
X = np.array(X)
Z = np.array(Z)

# Encode labels
encoder = LabelEncoder()
Z = encoder.fit_transform(Z)  # Convert labels to numbers
Z = to_categorical(Z, num_classes=5)  # Convert to one-hot encoding

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, Z, test_size=0.2, random_state=42)

# Model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(5,5), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=96, kernel_size=(5,5), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=96, kernel_size=(5,5), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())



model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(5, activation="softmax"))

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(x_train)

# Train model
batch_size = 128
epochs = 60

History = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    verbose=1,
                    steps_per_epoch=x_train.shape[0] // batch_size)

# Model Summary
model.summary()

model.save("flower_classification_model.keras") 