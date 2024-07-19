from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from keras.layers import Input

class CNN:
    @staticmethod
    def build(width, height, depth, total_classes, saved_weights_path=None):
        # Initialize the Model
        model = Sequential()

        # Define the input shape
        input_shape = (height, width, depth)
        
        # First CONV => RELU => POOL Layer
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second CONV => RELU => POOL Layer
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Third CONV => RELU => POOL Layer
        model.add(Conv2D(100, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Softmax classifier
        model.add(Dense(total_classes))
        model.add(Activation("softmax"))

        # Load pre-trained weights if provided
        if saved_weights_path is not None:
            model.load_weights(saved_weights_path)
        
        return model
