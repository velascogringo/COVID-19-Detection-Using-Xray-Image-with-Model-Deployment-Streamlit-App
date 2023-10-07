import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Data Augmentation for Training Data
train_datagen = ImageDataGenerator(
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_data = train_datagen.flow_from_directory(
    'COVID Dataset\\train',
    target_size=(256, 256),
    batch_size=16,
    class_mode='binary'
)
train_data.class_indices

# Data Augmentation for Test Data
test_datagen = ImageDataGenerator(rescale=1/255)
test_data = test_datagen.flow_from_directory(
    'COVID Dataset\\test',
    target_size=(256, 256),
    batch_size=16,
    class_mode='binary'
)
test_data.class_indices

# Define the Sequential Model
model = Sequential()

# Add Convolutional and Pooling Layers
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(256, 256, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

# Flatten the Model
model.add(Flatten())

# Add Fully Connected Layers
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display Model Summary
model.summary()

# Train the Model
model.fit_generator(train_data, steps_per_epoch=30, epochs=10, validation_data=test_data)

# Evaluate the Model on the Test Data
evaluation = model.evaluate(test_data)
loss = evaluation[0]
accuracy = evaluation[1]

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Evaluate the Model on the Training Data
training_accuracy = model.evaluate(train_data)[1]
print(f"Training Accuracy: {training_accuracy * 100:.2f}%")

# Evaluate the Model on the Test Data
test_accuracy = model.evaluate(test_data)[1]
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# Test the Model with a Single Image
path = "C:\\Users\\Hp\\Downloads\\Covid19_Model_Testing\\test\\Normal\\0117.jpeg"

img = image.load_img(path, target_size=(256, 256, 3))
img = image.img_to_array(img) / 255.0

plt.imshow(img)
img.shape

# Reshape the Image for Prediction
img = img.reshape((1, 256, 256, 3))

# Predict using the Model
prediction = model.predict(img)

# Display the Prediction Result
if prediction[0][0] > 0.5:
    print("Prediction: COVID-19 POSITIVE")
else:
    print("Prediction: NORMAL CHEST XRAY")

# Save the Model
model.save('covid_detection_model.h5')
