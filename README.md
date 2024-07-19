# Handwritten Digit Recognition using Deep Learning
This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset using Keras with TensorFlow.

## Project Structure
- CNN_MNIST.py: Main script for training, evaluating, and predicting using the CNN model.
- neural_network.py: Defines the CNN architecture.

## Installation
- Install dependencies: pip install numpy pandas scikit-learn keras tensorflow opencv-python
- Ensure the directory structure:
Handwritten-Digit-Recognition/
├── CNN_MNIST.py
├── cnn/
│   ├── __init__.py
│   └── neural_network.py
└── README.md
__init__.py should contain: from .neural_network import CNN

## Usage
Command-Line Arguments
- -s or --save_model: Save model weights (use 1 to enable).
- -l or --load_model: Load model weights (use 1 to enable).
- -w or --save_weights: Path for saving/loading weights.

-Examples
  - Train the Model: python CNN_MNIST.py --save_model 1 --save_weights weights.h5
  - Load and Evaluate: python CNN_MNIST.py --load_model 1 --save_weights weights.h5

## Code Overview
- CNN_MNIST.py:
Loads and preprocesses the MNIST dataset.
Defines and compiles the CNN model.
Trains, evaluates, and saves the model.
Displays sample predictions.
- neural_network.py:
Defines the CNN architecture with convolutional, pooling, and fully connected layers.
Loads pre-trained weights if provided.

## Troubleshooting
- Import Errors: Ensure all required libraries are installed.
- Infinite Running: Check dataset loading and training parameters.

## License
MIT License - see the LICENSE file for details.

## Acknowledgements
- MNIST Dataset
- Keras Documentation
- TensorFlow Documentation
