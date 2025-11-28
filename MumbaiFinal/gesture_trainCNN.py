import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set seeds
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

train_path = r'D:\gesture\train'
test_path = r'D:\gesture\test'

# Data generators
train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
).flow_from_directory(
    directory=train_path, target_size=(64,64),
    class_mode='categorical', batch_size=16, shuffle=True
)

test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
).flow_from_directory(
    directory=test_path, target_size=(64,64),
    class_mode='categorical', batch_size=16, shuffle=True
)

# -------------------------------
# Model Architecture
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    BatchNormalization(),
    MaxPool2D(2,2),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')  # FIXED
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
history = model.fit(
    train_batches,
    validation_data=test_batches,
    epochs=20,
    callbacks=[reduce_lr, early_stop]
)

scores = model.evaluate(test_batches)
print(f"Test Loss: {scores[0]:.4f}, Test Accuracy: {scores[1]*100:.2f}%")

model.save('best_model_gesture.h5')
print("Model saved successfully as best_model_gesture.h5")

# Prediction word dictionary
word_dict = {0:'One', 1:'Two', 2:'Three', 3:'Four', 4:'Five'}

# Sample prediction
imgs, labels = next(test_batches)
preds = model.predict(imgs)

for i in range(5):
    actual = word_dict[np.argmax(labels[i])]
    predicted = word_dict[np.argmax(preds[i])]

    print(f"Image {i+1}")
    print(" Predicted:", predicted)
    print(" Actual   :", actual)
    print(" Probabilities:", preds[i])
    print()

