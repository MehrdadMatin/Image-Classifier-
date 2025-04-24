import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data() # Load CIFAR-10 dataset
training_images, testing_images = training_images / 255.0, testing_images / 255.0 # Normalize pixel values to be between 0 and 1

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'] # CIFAR-10 classes

# for i in range(16):
#     plt.subplot(4, 4, i + 1) # 4 rows, 4 columns, i+1 index
#     plt.xticks([]) # Hide x ticks
#     plt.yticks([]) # Hide y ticks
#     plt.imshow(training_images[i], cmap=plt.cm.binary) # Display image
#     plt.xlabel(class_names[training_labels[i][0]]) # Set x label to class name
    
# plt.show() # Show the plot

training_images = training_images[:20000] # Use only 20,000 training images
training_labels = training_labels[:20000] # Use only 20,000 training labels
testing_images = testing_images[:4000] # Use only 4,000 testing images
testing_labels = testing_labels[:4000] # Use only 4,000 testing labels

# Uncomment the following lines to create and train a new model
model = models.Sequential() # Create a sequential model
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # Add a convolutional layer with 32 filters, 3x3 kernel size, ReLU activation function, and input shape of (32, 32, 3) - ConvLayer filters for features
model.add(layers.MaxPooling2D((2, 2))) # Add a max pooling layer with 2x2 pool size
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Add another convolutional layer with 64 filters and 3x3 kernel size
model.add(layers.MaxPooling2D((2, 2))) # Add another max pooling layer with 2x2 pool size
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Add another convolutional layer with 64 filters and 3x3 kernel size
model.add(layers.Flatten()) # Flatten the output from the previous layer
model.add(layers.Dense(64, activation='relu')) # Add a dense layer with 64 units and ReLU activation function
model.add(layers.Dense(10, activation='softmax')) # Add a dense layer with 10 units (one for each class) 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # Compile the model with Adam optimizer, sparse categorical crossentropy loss function, and accuracy metric

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels)) # Train the model for 10 epochs with validation data

loss, accuracy = model.evaluate(testing_images, testing_labels) # Evaluate the model on the testing data
print(f"Loss: {loss}") # Print the loss
print(f"Accuracy: {accuracy}") # Print the accuracy

model.save('image_classifier.keras') # Save the model to a file

# Uncomment the following line to load the model from the file
# model = models.load_model('image_classifier.keras') # Load the model from the file

img = cv.imread('path/to/image.jpg') # TODO: replace with your image file path
img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # Convert the image from BGR to RGB

plt.imshow(img, cmap=plt.cm.binary) # Display the image

prediction = model.predict(np.array([img]) / 255) # Make a prediction on the image
index = np.argmax(prediction) # Get the index of the class with the highest probability
print(f"Prediction is {class_names[index]}") # Print the predicted class name

plt.show() # Show the plot

    
    