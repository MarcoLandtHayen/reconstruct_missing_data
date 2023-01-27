import numpy as np
import tensorflow as tf
import tensorflow.keras.initializers as tfi
import tensorflow.keras.regularizers as tfr

from matplotlib import pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model


# Suppress Tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Define function to compute relevance map for single sample:
def compute_single_relevance_map(input_sample, patch_size, model):
    """Compute relevance map for single sample, divided into squared patches, from given model.

    Parameters
    ----------
    input_sample: numpy.ndarray
        Single input sample, dimension (1, latitude, longitude, 1), where the last dimension specifies the number of channels, here: 1.
    patch_size: int
        Size of edges of squared patches, that the sample will divided to.
    model: tensorflow.model
        Pre-trained model to get predictions from.
        
    Returns
    -------
    numpy.ndarray
        Heat map of relative loss reduction of each patch, when added. Total relative loss sums up to one. Dimensions equal original input sample's latitude and longitude.

    """
    
    ## Get parameters for patches, enumerated line-by-line, from left to right, from top to bottom, starting with ONE.

    # Get number of patches in lat and lon directions, respectively:
    n_lat = int(input_sample.shape[1] / patch_size)
    n_lon = int(input_sample.shape[2] / patch_size)

    # Obtain number of patches:
    n_patches = int(n_lat * n_lon)

    # Create list of patch indices:
    patch_indices = list(np.arange(n_patches))
    
    # Create empty sample of just ZEROs:
    empty_sample = np.zeros((1, input_sample.shape[1], input_sample.shape[2]))

    ## Create patches:

    # Initialize storage for patches. Dimension (# of patches, latitude, longitude)
    patches = np.zeros((len(patch_indices), input_sample.shape[1], input_sample.shape[2]))

    # Run over list of patch indices:
    for n in range(len(patch_indices)):

        # Get current patch index:
        patch_index = patch_indices[n]

        # Get x and y coordinate from current patch index:
        y = patch_index // n_lon
        x = patch_index % n_lon    

        # Store mask for current patch:
        patches[n,int(y*patch_size):int((y+1)*patch_size),int(x*patch_size):int((x+1)*patch_size)] = 1 
    
    # Expand dimensions of patches: Have last dimension for channel (=1), to match requirements for CNN inputs.
    patches_extended = np.expand_dims(patches, axis=-1)

    # Create input samples from first validation sample:
    patchy_input = patches_extended * input_sample
    
    # Get mean state on empty sample as input:
    mean_state = model.predict(empty_sample)
    
    # Get prediction from complete sample as input:
    full_pred = model.predict(input_sample)

    # Get model predictions on patchy inputs:
    patchy_pred = model.predict(patchy_input)

    # Compute mean state loss from prediction on empty sample compared to target (= complete input sample):
    mean_state_loss = np.mean((mean_state[:,:,:,0] - input_sample[0,:,:,0])**2)
    
    # Compute min loss from prediction on complete sample compared to target (= complete input sample):
    min_loss = np.mean((full_pred[:,:,:,0] - input_sample[0,:,:,0])**2)
    
    # Compute loss of patchy predictions compared to targets (= complete input sample):
    patchy_loss = np.mean((patchy_pred[:,:,:,0] - input_sample[0,:,:,0])**2,axis=(1,2))
    
    # Initialize storage for patches ordered by decreasing relevance, absolute and relative loss reduction:
    patch_order = []
    abs_loss_reduction = []
    rel_loss_reduction = []

    # get index for patch leading to lowest loss, when adding:
    min_index = np.argsort(patchy_loss)[0]

    # Save index of first patch, leading to lowest loss:
    patch_order.append(patch_indices[min_index])

    # Save absolute loss reduction, when adding this patch:
    abs_loss_reduction.append(mean_state_loss - patchy_loss[min_index])

    # Save loss reduction relative to the difference of mean state loss and min. loss, when adding this patch:
    rel_loss_reduction.append((mean_state_loss - patchy_loss[min_index]) / (mean_state_loss - min_loss))

    # Fix the previously identified patch with lowest reconstruction loss, as new base patch:
    base_patch = patches[min_index]
    
    for i in range(5):
    
        ## Create new patches:

        # Remove previously selected patch from list of patch indices:
        patch_indices.remove(patch_indices[min_index])

        # Initialize storage for patches by repeating base patch as often as we have remaining patches.
        # Dimensions: (# of remaining patches, latitude, longitude)
        patches = np.repeat(np.expand_dims(base_patch,axis=0),len(patch_indices), axis=0)

        # Run over list of remaining patch indices:
        for n in range(len(patch_indices)):

            # Get current patch index:
            patch_index = patch_indices[n]

            # Get x and y coordinate from current patch index:
            y = patch_index // n_lon
            x = patch_index % n_lon    

            # Store mask for current patch:
            patches[n,int(y*patch_size):int((y+1)*patch_size),int(x*patch_size):int((x+1)*patch_size)] = 1 

        # Create base input from base patch:
        base_input = (np.expand_dims(base_patch, axis=-1) * input_sample)

        # Get model prediction on base input:
        base_pred = model.predict(base_input)

        # Expand dimensions of patches: Have last dimension for channel (=1), to match requirements for CNN inputs.
        patches_extended = np.expand_dims(patches, axis=-1)

        # Create input samples from first validation sample:
        patchy_input = patches_extended * input_sample

        # Get model predictions on patchy inputs:
        patchy_pred = model.predict(patchy_input)

        # Compute loss from prediction on base sample compared to target (= complete input sample):
        base_loss = np.mean((base_pred[:,:,:,0] - input_sample[0,:,:,0])**2)

        # Compute loss (mean squared error) of patchy predictions compared to targets (= complete input sample):
        patchy_loss = np.mean((patchy_pred[:,:,:,0] - input_sample[0,:,:,0])**2,axis=(1,2))

        # get index for patch leading to lowest loss, when adding:
        min_index = np.argsort(patchy_loss)[0]

        # Save index of first patch, leading to lowest loss:
        patch_order.append(patch_indices[min_index])

        # Save absolute loss reduction, when adding this patch:
        abs_loss_reduction.append(base_loss - patchy_loss[min_index])

        # Save loss reduction relative to the difference of mean state loss and min. loss, when adding this patch:
        rel_loss_reduction.append((base_loss - patchy_loss[min_index]) / (mean_state_loss - min_loss))

        # Fix the previously identified patch with lowest reconstruction loss, as new base patch:
        base_patch = patches[min_index]

    ## Post-processing of patch order, in combination with rel. loss reduction.
    ## Aim to have a heat map with original size from input samples in latitude and longitude.
    ## Grid points for each patch get rel. loss reduction of individual patch as constant value.

    # Initialize storage:
    rel_loss_reduction_map = np.zeros((input_sample.shape[1], input_sample.shape[2]))

    # Run over list containing patch order:
    for n in range(len(patch_order)):

        # Get current patch index:
        patch_index = patch_order[n]

        # Get x and y coordinate from current patch index:
        y = patch_index // n_lon
        x = patch_index % n_lon    

        # Store rel. loss reduction for current patch:
        rel_loss_reduction_map[int(y*patch_size):int((y+1)*patch_size),int(x*patch_size):int((x+1)*patch_size)] = rel_loss_reduction[n] 
    
    return rel_loss_reduction_map