import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate, Conv1D, Conv2D, MaxPool2D, UpSampling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.initializers as tfi
import tensorflow.keras.regularizers as tfr
from tensorflow.keras.utils import plot_model

# Suppress Tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def build_unet_4conv(input_shape, CNN_filters, CNN_kernel_size, learning_rate, loss_function):

    """Build and compile U-Net model with 4 convolutional layers.

    Parameters
    ----------
    input_shape: list
        Contains triple: (latitude, longitude, 1), where trailing 1 denotes the number of channels.
    CNN_filters: np.ndarray of int
        Contains 4 numbers to specify the number of filters in each of the 4 convolutions.
    CNN_kernel_size: int
        Specifies the kernel size, taken to be similar for each convolution.
    learning_rate: float
        Learning rate.
    loss_function: string
        Loss function, e.g. 'mse'.
        
    Returns
    -------
    Tensorflow model
        Compiled model.
    """
        
    # Add input layer according to specified input shape:
    inputs = Input(shape=input_shape)

    # Add CNN layers:
    cnn_1 = Conv2D(filters=CNN_filters[0], kernel_size=CNN_kernel_size, padding='same', activation='relu')(inputs)
    cnn_2 = Conv2D(filters=CNN_filters[0], kernel_size=CNN_kernel_size, padding='same', activation='relu')(cnn_1)

    # Add max. pooling (2x2):
    pool_1 = MaxPool2D(pool_size=(2,2))(cnn_2)

    # Add CNN layers:
    cnn_3 = Conv2D(filters=CNN_filters[1], kernel_size=CNN_kernel_size, padding='same', activation='relu')(pool_1)
    cnn_4 = Conv2D(filters=CNN_filters[1], kernel_size=CNN_kernel_size, padding='same', activation='relu')(cnn_3)

    # Add max. pooling (2x2):
    pool_2 = MaxPool2D(pool_size=(2,2))(cnn_4)

    # Add CNN layers:
    cnn_5 = Conv2D(filters=CNN_filters[2], kernel_size=CNN_kernel_size, padding='same', activation='relu')(pool_2)
    cnn_6 = Conv2D(filters=CNN_filters[2], kernel_size=CNN_kernel_size, padding='same', activation='relu')(cnn_5)

    # Add max. pooling (2x2):
    pool_3 = MaxPool2D(pool_size=(2,2))(cnn_6)

    # Add CNN layers:
    cnn_7 = Conv2D(filters=CNN_filters[3], kernel_size=CNN_kernel_size, padding='same', activation='relu')(pool_3)
    cnn_8 = Conv2D(filters=CNN_filters[3], kernel_size=CNN_kernel_size, padding='same', activation='relu')(cnn_7)

    # Add upsampling (2x2):
    up_1 = UpSampling2D(size=(2,2))(cnn_8)

    # Add Concatenation:
    concat_1 = concatenate([up_1,cnn_6])

    # Add CNN layers:
    cnn_9 = Conv2D(filters=CNN_filters[2], kernel_size=CNN_kernel_size, padding='same', activation='relu')(concat_1)
    cnn_10 = Conv2D(filters=CNN_filters[2], kernel_size=CNN_kernel_size, padding='same', activation='relu')(cnn_9)

    # Add upsampling (2x2):
    up_2 = UpSampling2D(size=(2,2))(cnn_10)

    # Add Concatenation:
    concat_2 = concatenate([up_2,cnn_4])

    # Add CNN layers:
    cnn_11 = Conv2D(filters=CNN_filters[1], kernel_size=CNN_kernel_size, padding='same', activation='relu')(concat_2)
    cnn_12 = Conv2D(filters=CNN_filters[1], kernel_size=CNN_kernel_size, padding='same', activation='relu')(cnn_11)

    # Add upsampling (2x2):
    up_3 = UpSampling2D(size=(2,2))(cnn_12)

    # Add Concatenation:
    concat_3 = concatenate([up_3,cnn_2])

    # Add CNN layers:
    cnn_13 = Conv2D(filters=CNN_filters[0], kernel_size=CNN_kernel_size, padding='same', activation='relu')(concat_3)
    cnn_14 = Conv2D(filters=CNN_filters[0], kernel_size=CNN_kernel_size, padding='same', activation='relu')(cnn_13)

    # Add final convolution: Only ONE filter with filter size ONE!
    output = Conv2D(filters=1, kernel_size=1, padding='same', activation='relu')(cnn_14)

    # Define and compile model :
    model = Model(inputs, output, name='U-Net')

    # Compile model with desired loss function:
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=([]))

    return model