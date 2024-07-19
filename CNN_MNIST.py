import numpy as np
import argparse
import cv2
from cnn import CNN  # Importing CNN from the cnn package
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Parse the Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save_model", type=int, default=-1)
ap.add_argument("-l", "--load_model", type=int, default=-1)
ap.add_argument("-w", "--save_weights", type=str)
args = vars(ap.parse_args())

# Read/Download MNIST Dataset
print('Loading MNIST Dataset...')
dataset = fetch_openml('mnist_784')

# Convert dataset.data to numpy array and reshape
mnist_data = dataset.data.values.reshape((dataset.data.shape[0], 28, 28, 1))

# Normalize data to range [0, 1]
mnist_data = mnist_data / 255.0

# Use a smaller subset of the data for initial testing
mnist_data, _, mnist_labels, _ = train_test_split(mnist_data, dataset.target.astype("int"), train_size=0.1)

# Divide data into testing and training sets.
train_img, test_img, train_labels, test_labels = train_test_split(mnist_data, mnist_labels, test_size=0.1)

# Transform training and testing labels to categorical
total_classes = 10  # 0 to 9 labels
train_labels = to_categorical(train_labels, total_classes)
test_labels = to_categorical(test_labels, total_classes)

print("Data Loaded and Preprocessed Successfully")
print("Training Images Shape:", train_img.shape)
print("Test Images Shape:", test_img.shape)
print("Training Labels Shape:", train_labels.shape)
print("Test Labels Shape:", test_labels.shape)

# Model Definition and Compilation
print('\nCompiling model...')
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
clf = CNN.build(width=28, height=28, depth=1, total_classes=10, saved_weights_path=args["save_weights"] if args["load_model"] > 0 else None)
clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

print("Model Compiled Successfully")

# Callbacks for learning rate reduction and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Initially train and test the model; If weight saved already, load the weights using arguments.
b_size = 32  # Reduced Batch size
num_epoch = 2  # Further Reduced Number of epochs for initial testing
verb = 1  # Verbose

# If weights saved and argument load_model; Load the pre-trained model.
if args["load_model"] < 0:
    print('\nTraining the Model...')
    history = clf.fit(train_img, train_labels, batch_size=b_size, epochs=num_epoch, verbose=verb, validation_split=0.1, callbacks=[reduce_lr, early_stopping])
    
    # Evaluate accuracy and loss function of test data
    print('Evaluating Accuracy and Loss Function...')
    loss, accuracy = clf.evaluate(test_img, test_labels, batch_size=128, verbose=1)
    print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))

# Save the pre-trained model.
if args["save_model"] > 0:
    print('Saving weights to file...')
    clf.save_weights(args["save_weights"], overwrite=True)

print("Training and Evaluation Completed Successfully")

# Show the images using OpenCV and making random selections.
for num in np.random.choice(np.arange(0, len(test_labels)), size=(5,)):
    # Predict the label of digit using CNN.
    probs = clf.predict(test_img[np.newaxis, num])
    prediction = probs.argmax(axis=1)

    # Resize the Image to 100x100 from 28x28 for better view.
    image = (test_img[num] * 255).astype("uint8")
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, str(prediction[0]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Show and print the Actual Image and Predicted Label Value
    print('Predicted Label: {}, Actual Value: {}'.format(prediction[0], np.argmax(test_labels[num])))
    cv2.imshow('Digits', image)
    cv2.waitKey(0)

# EOC
