import torch
import numpy as np
import matplotlib.pyplot as plt


def flatten_batch(tensor_batch):
    """
    Flattens a batch of 4D tensors into a 2D array.

    Args:
        tensor_batch (torch.Tensor): A 4D tensor of shape (N, C, H, W) where
                                     N is the batch size,
                                     C is the number of channels,
                                     H is the height,
                                     W is the width.

    Returns:
        numpy.ndarray: A 2D array of shape (N*H*W, C) where each row corresponds
                       to a flattened version of the input tensor's spatial dimensions.
    """
    if tensor_batch.dim() == 1:
        return tensor_batch#.numpy()
    C = tensor_batch.shape[1]
    permute_dims = list(range(len(tensor_batch.shape)))
    permute_dims.append(permute_dims.pop(1))  # Move the Channel dimension to the end  # TODO: This could be more general

    tensor_flat = tensor_batch.permute(*permute_dims).reshape(-1, C)
    return tensor_flat#.numpy()


def unflatten_batch(flat_array, original_shape):
    """
    Reconstruct original tensor from a flattened array obtained from 'flatten_batch'.
    Assumes Channel dimension in position 1.

    Args:
        flat_array (numpy.ndarray): 2D Array of shape (N*..., C).
        original_shape (tuple): Original tensor shape before flatten, e.g. (N, C, H, W)

    Returns:
        torch.Tensor: Reconstructed tensor.
    """
    if len(original_shape) == 1:
        return torch.from_numpy(flat_array)

    # Reconstruct the original shape
    C = original_shape[1]
    spatial_shape = original_shape[:1] + original_shape[2:]  # quitamos el canal
    num_elements = 1
    for dim in spatial_shape:
        num_elements *= dim

    assert flat_array.shape[0] == num_elements, (
        f"Element number does not match: {flat_array.shape[0]} != {num_elements}"
    )
    assert flat_array.shape[1] == C, (
        f"Channel number does not match: {flat_array.shape[1]} != {C}"
    )

    # Convert back to tensor and rearange dims
    tensor = torch.from_numpy(flat_array)
    new_shape = spatial_shape + (C,)  # canales al final
    tensor = tensor.reshape(*new_shape)

    # Channel dimension back in position 1
    num_dims = len(original_shape)
    permute_back = [0]  # N
    permute_back.insert(1, num_dims - 1)
    permute_back += [i for i in range(1, num_dims - 1)]  # remaining dims

    tensor = tensor.permute(*permute_back)
    return tensor


def unflatten_pixels(pixel_array, batch_size, height, width):
    """
    Reshapes a flattened pixel array into a 4D tensor with dimensions suitable for image processing.

    This function should be replaced with 'unflatten_batch' in the future.

    Args:
        pixel_array (numpy.ndarray): The input pixel array with shape (batch_size * height * width, C).
        batch_size (int): The number of images in the batch.
        height (int): The height of each image.
        width (int): The width of each image.

    Returns:
        torch.Tensor: A 4D tensor with shape (batch_size, C, height, width).
    """
    C = pixel_array.shape[1]
    tensor_reshaped = torch.tensor(pixel_array, dtype=torch.float32).reshape(batch_size, height, width, C)
    return tensor_reshaped.permute(0, 3, 1, 2)


def unflatten_set_sizes(sets_array, batch_size, height, width):
    """
    Reshapes a flattened output sets array into a 4D tensor with dimensions suitable for image processing.

    Args:
        sets_array (numpy.ndarray): The CP ouput sets array with shape (batch_size * height * width, Classes).
        batch_size (int): The number of images in the batch.
        height (int): The height of each image.
        width (int): The width of each image.

    Returns:
        torch.Tensor: A 4D tensor with shape (batch_size, C, height, width).
    """
    setsizes = sets_array.sum(axis=1)
    C = 1  # Only one channel for setsize
    tensor_reshaped = torch.tensor(setsizes, dtype=torch.float32).reshape(batch_size, height, width, C)
    return tensor_reshaped.permute(0, 3, 1, 2)


def view_classify(img, ps, version="MNIST"):
    """
    Function for viewing an image and its predicted classes.

    Parameters:
    img (torch.Tensor): The image to be displayed.
    ps (torch.Tensor): The predicted class probabilities.
    version (str): The dataset version, either "MNIST" or "Fashion". Default is "MNIST".

    Returns:
    None
    """
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
