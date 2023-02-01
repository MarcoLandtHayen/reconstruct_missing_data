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


# Define function to compute relevance map for single sample in a more compact way.
# This function uses more memory and might crash for multi-sample processing.
def compute_single_relevance_map(input_sample, patch_size, model, max_patch_num, max_acc_rel_loss_reduction):
    """Compute relevance map for single sample, divided into squared patches, from given model.

    Parameters
    ----------
    input_sample: numpy.ndarray
        Single input sample, dimension (1, latitude, longitude, 1), where the last dimension specifies the number of channels, here: 1.
    patch_size: int
        Size of edges of squared patches, that the sample will divided to.
    model: tensorflow.model
        Pre-trained model to get predictions from.
    max_patch_num: int
        Stop adding patches after reaching this specified number of maximum patches to include.
        If set to -1, include ALL patches.
    max_acc_rel_loss_reduction: float
        Or stop adding patches after accumulated rel. loss reduction exceeds this specified threshold.
        
    Returns
    -------
    numpy.ndarray
        Heat map of relative loss reduction of each patch, when added. Total relative loss sums up to one, if include all patches. 
        Dimensions equal original input sample's latitude and longitude.
    
    numpy.ndarrays patch_order, abs_loss_reduction, rel_loss_reduction, acc_rel_loss_reduction
        Patch numbers ordered by decreasing relevance, absolute and relative loss reduction, and accumulated rel. loss reduction

    """
    
    ## Get parameters for patches, enumerated line-by-line, from left to right, from top to bottom, starting with ZERO.

    # Get number of patches in lat and lon directions, respectively:
    n_lat = int(input_sample.shape[1] / patch_size)
    n_lon = int(input_sample.shape[2] / patch_size)

    # Obtain total number of patches:
    n_patches = int(n_lat * n_lon)
    
    # Check for maximum number of desired patches: If given as -1, set to total number of patches.
    if max_patch_num == -1:
        max_patch_num = n_patches

    # Create list of patch indices:
    patch_indices = list(np.arange(n_patches))
    
    # Create empty sample of just ZEROs:
    empty_sample = np.zeros((1, input_sample.shape[1], input_sample.shape[2]))

    ## Create patches:

    # Initialize storage for patches as boolean array. Dimension (# of patches, latitude, longitude)
    patches = (np.zeros((len(patch_indices), input_sample.shape[1], input_sample.shape[2])) != 0)

    # Run over list of patch indices:
    for n in range(len(patch_indices)):

        # Get current patch index:
        patch_index = patch_indices[n]

        # Get x and y coordinate from current patch index:
        y = patch_index // n_lon
        x = patch_index % n_lon    

        # Store mask for current patch:
        patches[n,int(y*patch_size):int((y+1)*patch_size),int(x*patch_size):int((x+1)*patch_size)] = True 
    
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
    
    # Initialize storage for patches ordered by decreasing relevance, absolute and relative loss reduction, 
    # and accumulated rel. loss reduction:
    patch_order = []
    abs_loss_reduction = []
    rel_loss_reduction = []
    acc_rel_loss_reduction = []

    # get index for patch leading to lowest loss, when adding:
    min_index = np.argsort(patchy_loss)[0]

    # Save index of first patch, leading to lowest loss:
    patch_order.append(patch_indices[min_index])

    # Save absolute loss reduction, when adding this patch:
    abs_loss_reduction.append(mean_state_loss - patchy_loss[min_index])

    # Save loss reduction relative to the difference of mean state loss and min. loss, when adding this patch:
    rel_loss_reduction.append((mean_state_loss - patchy_loss[min_index]) / (mean_state_loss - min_loss))
    
    # Save accumulated rel. loss reduction, for first patch it equals the usual rel. loss reduction:
    acc_rel_loss_reduction.append(rel_loss_reduction[0])

    # Fix the previously identified patch with lowest reconstruction loss, as new base patch:
    base_patch = patches[min_index]
    
    ## Run over the remaining patches, check, if stopping criterions are fulfilled:
    for i in range(n_patches-1):

        # Check for maximum number of patches to include and threshold for accumulated rel. loss reduction:
        if (len(patch_order) >= max_patch_num) | (acc_rel_loss_reduction[-1] >= max_acc_rel_loss_reduction):
            break
        else:        
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
                patches[n,int(y*patch_size):int((y+1)*patch_size),int(x*patch_size):int((x+1)*patch_size)] = True 

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
            
            # Save accumulated rel. loss reduction:
            acc_rel_loss_reduction.append(acc_rel_loss_reduction[-1]+rel_loss_reduction[-1])

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

    ## Convert lists to np.arrays of unified maximum length (n_patches), for convenient post-processing:
    
    # Initialize storages, want to recognize unfilled entries as NaN, to avoid misunderstanding:
    patch_order_array = np.zeros(n_patches)+float('nan')
    abs_loss_reduction_array = np.zeros(n_patches)+float('nan')
    rel_loss_reduction_array = np.zeros(n_patches)+float('nan')
    acc_rel_loss_reduction_array = np.zeros(n_patches)+float('nan')
    
    # Fill lists' content into arrays:
    patch_order_array[:len(patch_order)]=patch_order
    abs_loss_reduction_array[:len(patch_order)]=abs_loss_reduction
    rel_loss_reduction_array[:len(patch_order)]=rel_loss_reduction
    acc_rel_loss_reduction_array[:len(patch_order)]=acc_rel_loss_reduction
            
    return rel_loss_reduction_map, patch_order_array, abs_loss_reduction_array, rel_loss_reduction_array, acc_rel_loss_reduction_array




# Define function to compute relevance map for single sample in a memory-saving way.
# This function has longer run-time, but allows multi-sample processing.
def compute_single_relevance_map_slow(input_sample, patch_size, model, max_patch_num, max_acc_rel_loss_reduction):
    """Compute relevance map for single sample, divided into squared patches, from given model.

    Parameters
    ----------
    input_sample: numpy.ndarray
        Single input sample, dimension (1, latitude, longitude, 1), where the last dimension specifies the number of channels, here: 1.
    patch_size: int
        Size of edges of squared patches, that the sample will divided to.
    model: tensorflow.model
        Pre-trained model to get predictions from.
    max_patch_num: int
        Stop adding patches after reaching this specified number of maximum patches to include.
        If set to -1, include ALL patches.
    max_acc_rel_loss_reduction: float
        Or stop adding patches after accumulated rel. loss reduction exceeds this specified threshold.
        
    Returns
    -------
    numpy.ndarray
        Heat map of relative loss reduction of each patch, when added. Total relative loss sums up to one, if include all patches. 
        Dimensions equal original input sample's latitude and longitude.
    
    numpy.ndarrays patch_order, abs_loss_reduction, rel_loss_reduction, acc_rel_loss_reduction
        Patch numbers ordered by decreasing relevance, absolute and relative loss reduction, and accumulated rel. loss reduction

    """
    
    ## Get parameters for patches, enumerated line-by-line, from left to right, from top to bottom, starting with ZERO.

    # Get number of patches in lat and lon directions, respectively:
    n_lat = int(input_sample.shape[1] / patch_size)
    n_lon = int(input_sample.shape[2] / patch_size)

    # Obtain total number of patches:
    n_patches = int(n_lat * n_lon)
    
    # Check for maximum number of desired patches: If given as -1, set to total number of patches.
    if max_patch_num == -1:
        max_patch_num = n_patches

    # Create list of patch indices:
    patch_indices = list(np.arange(n_patches))
    
    # Create empty sample of just ZEROs:
    empty_sample = np.zeros((1, input_sample.shape[1], input_sample.shape[2]))
    
    # Get mean state on empty sample as input:
    mean_state = model.predict(empty_sample)
    
    # Get prediction from complete sample as input:
    full_pred = model.predict(input_sample)
    
    # Compute mean state loss from prediction on empty sample compared to target (= complete input sample):
    mean_state_loss = np.mean((mean_state[:,:,:,0] - input_sample[0,:,:,0])**2)
    
    # Compute min loss from prediction on complete sample compared to target (= complete input sample):
    min_loss = np.mean((full_pred[:,:,:,0] - input_sample[0,:,:,0])**2)

    ## Process patches, one-by-one, to save memory:

    # Initialize storage for patchy loss:
    patchy_loss_all = []
    
    # Run over list of patch indices:
    for n in range(len(patch_indices)):

        # Initialize storage for current patch as boolean array. Dimension (1, latitude, longitude)
        patch = (np.zeros((1, input_sample.shape[1], input_sample.shape[2])) != 0)

        # Get current patch index:
        patch_index = patch_indices[n]

        # Get x and y coordinate from current patch index:
        y = patch_index // n_lon
        x = patch_index % n_lon    

        # Create mask for current patch:
        patch[0,int(y*patch_size):int((y+1)*patch_size),int(x*patch_size):int((x+1)*patch_size)] = True 
    
        # Expand dimensions of patch: Have last dimension for channel (=1), to match requirements for CNN inputs.
        patch_extended = np.expand_dims(patch, axis=-1)

        # Create input applying patch to given input sample:
        patchy_input = patch_extended * input_sample

        # Get model predictions on patchy input:
        patchy_pred = model.predict(patchy_input)
 
        # Compute loss of patchy prediction compared to target (= complete input sample):
        patchy_loss = np.mean((patchy_pred[:,:,:,0] - input_sample[0,:,:,0])**2,axis=(1,2))

        # Store patchy loss for current patch:
        patchy_loss_all.append(patchy_loss[0])
        
    # Convert list containing patchy loss for all patches into numpy array:
    patchy_loss_all = np.array(patchy_loss_all)    
    
    # Initialize storage for patches ordered by decreasing relevance, absolute and relative loss reduction, 
    # and accumulated rel. loss reduction:
    patch_order = []
    abs_loss_reduction = []
    rel_loss_reduction = []
    acc_rel_loss_reduction = []

    # get index for patch leading to lowest loss, when adding:
    min_index = np.argsort(patchy_loss_all)[0]

    # Save index of first patch, leading to lowest loss:
    patch_order.append(patch_indices[min_index])

    # Save absolute loss reduction, when adding this patch:
    abs_loss_reduction.append(mean_state_loss - patchy_loss_all[min_index])

    # Save loss reduction relative to the difference of mean state loss and min. loss, when adding this patch:
    rel_loss_reduction.append((mean_state_loss - patchy_loss_all[min_index]) / (mean_state_loss - min_loss))
    
    # Save accumulated rel. loss reduction, for first patch it equals the usual rel. loss reduction:
    acc_rel_loss_reduction.append(rel_loss_reduction[0])

    ## Fix the previously identified patch with lowest reconstruction loss, as new base patch.
    ## Therefore rebuild corresponding batch from found min_index:
    
    # Initialize storage for new base patch as boolean array. Dimension (1, latitude, longitude)
    base_patch = (np.zeros((1, input_sample.shape[1], input_sample.shape[2])) != 0)
    
    # Get x and y coordinate for patch that corresponds to min_index:
    y = patch_indices[min_index] // n_lon
    x = patch_indices[min_index] % n_lon    

    # Create mask for base patch:
    base_patch[0,int(y*patch_size):int((y+1)*patch_size),int(x*patch_size):int((x+1)*patch_size)] = True 
    
    # Create base input from base patch and given input sample:
    base_input = (np.expand_dims(base_patch, axis=-1) * input_sample)

    # Get model prediction on base input:
    base_pred = model.predict(base_input)
    
    # Compute loss from prediction on base sample compared to target (= complete input sample):
    base_loss = np.mean((base_pred[:,:,:,0] - input_sample[0,:,:,0])**2)
            
    ## Run over the remaining patches, check, if stopping criterions are fulfilled:
    for i in range(n_patches-1):

        # Check for maximum number of patches to include and threshold for accumulated rel. loss reduction:
        if (len(patch_order) >= max_patch_num) | (acc_rel_loss_reduction[-1] >= max_acc_rel_loss_reduction):
            break
        else:        
            ## Create new patches:

            # Remove previously selected patch from list of patch indices:
            patch_indices.remove(patch_indices[min_index])

            ## Process remaining patches, one-by-one, to save memory:

            # Initialize storage for patchy loss:
            patchy_loss_all = []
            
            # Run over list of remaining patch indices:
            for n in range(len(patch_indices)):            
            
                # Initialize storage for current patch by copying former base patch.
                # Dimensions: (1, latitude, longitude)
                patch = np.copy(base_patch)

                # Get current patch index:
                patch_index = patch_indices[n]

                # Get x and y coordinate from current patch index:
                y = patch_index // n_lon
                x = patch_index % n_lon    

                # Store mask for current patch:
                patch[0,int(y*patch_size):int((y+1)*patch_size),int(x*patch_size):int((x+1)*patch_size)] = True 

                # Expand dimensions of patch: Have last dimension for channel (=1), to match requirements for CNN inputs.
                patch_extended = np.expand_dims(patch, axis=-1)

                # Create input sample from current patch and given input sample:
                patchy_input = patch_extended * input_sample

                # Get model prediction on patchy inputs:
                patchy_pred = model.predict(patchy_input)

                # Compute loss (mean squared error) of patchy prediction compared to target (= complete input sample):
                patchy_loss = np.mean((patchy_pred[:,:,:,0] - input_sample[0,:,:,0])**2,axis=(1,2))

                # Store patchy loss for current patch:
                patchy_loss_all.append(patchy_loss[0])
        
            # Convert list containing patchy loss for all remaining patches into numpy array:
            patchy_loss_all = np.array(patchy_loss_all)   
            
            # get index for patch leading to lowest loss, when adding:
            min_index = np.argsort(patchy_loss_all)[0]

            # Save index of first patch, leading to lowest loss:
            patch_order.append(patch_indices[min_index])

            # Save absolute loss reduction, when adding this patch:
            abs_loss_reduction.append(base_loss - patchy_loss_all[min_index])

            # Save loss reduction relative to the difference of mean state loss and min. loss, when adding this patch:
            rel_loss_reduction.append((base_loss - patchy_loss_all[min_index]) / (mean_state_loss - min_loss))
            
            # Save accumulated rel. loss reduction:
            acc_rel_loss_reduction.append(acc_rel_loss_reduction[-1]+rel_loss_reduction[-1])
            
            ## Fix the previously identified patch with lowest reconstruction loss, as new base patch.
            ## Therefore rebuild corresponding batch from found min_index:
            
            # Get x and y coordinate for patch index that corresponds to min_index:
            y = patch_indices[min_index] // n_lon
            x = patch_indices[min_index] % n_lon    

            # Create mask for base patch:
            base_patch[0,int(y*patch_size):int((y+1)*patch_size),int(x*patch_size):int((x+1)*patch_size)] = True 

            # Create base input from base patch and given input sample:
            base_input = (np.expand_dims(base_patch, axis=-1) * input_sample)

            # Get model prediction on base input:
            base_pred = model.predict(base_input)

            # Compute loss from prediction on base sample compared to target (= complete input sample):
            base_loss = np.mean((base_pred[:,:,:,0] - input_sample[0,:,:,0])**2)

    
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

    ## Convert lists to np.arrays of unified maximum length (n_patches), for convenient post-processing:
    
    # Initialize storages, want to recognize unfilled entries as NaN, to avoid misunderstanding:
    patch_order_array = np.zeros(n_patches)+float('nan')
    abs_loss_reduction_array = np.zeros(n_patches)+float('nan')
    rel_loss_reduction_array = np.zeros(n_patches)+float('nan')
    acc_rel_loss_reduction_array = np.zeros(n_patches)+float('nan')
    
    # Fill lists' content into arrays:
    patch_order_array[:len(patch_order)]=patch_order
    abs_loss_reduction_array[:len(patch_order)]=abs_loss_reduction
    rel_loss_reduction_array[:len(patch_order)]=rel_loss_reduction
    acc_rel_loss_reduction_array[:len(patch_order)]=acc_rel_loss_reduction
            
    return rel_loss_reduction_map, patch_order_array, abs_loss_reduction_array, rel_loss_reduction_array, acc_rel_loss_reduction_array



