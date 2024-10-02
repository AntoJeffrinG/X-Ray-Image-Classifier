import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

# Define image size and batch size
image_size = (224, 224)
batch_size = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for test/validation data
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    '/Users/jeffrin/Documents/vscode/ML group Project/test',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Load validation/test data
validation_generator = test_datagen.flow_from_directory(
    '/Users/jeffrin/Documents/vscode/ML group Project/train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Build the CNN model
model = Sequential()

# Convolution + Pooling Layer 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution + Pooling Layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution + Pooling Layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the results to feed into the dense layers
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output Layer (Binary Classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate the model on test data
loss, accuracy = model.evaluate(validation_generator)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
model.save('medical_image_classifier.h5')

#model usage
import cv2

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    
    if prediction[0] > 0.5:
        return 'Abnormal'
    else:
        return 'Normal'

# Example usage
result = predict_image('data/test/normal/sample_image.png')
print(f'Prediction: {result}')
