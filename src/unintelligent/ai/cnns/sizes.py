from typing import Tuple


def conv_output_size(input_size: int,
                     kernel_size: int,
                     padding_size: int = 0,
                     stride: int = 1) -> int:
    """Calculates the output size after applying a convolution in 1D.

    Args:
        input_size (int): input dimension
        kernel_size (int): kernel size
        padding_size (int, optional): padding size. Defaults to 0.
        stride (int, optional): kernel stride. Defaults to 1.

    Returns:
        int: output size after applying the convolution
    """

    return ((input_size - kernel_size + 2 * padding_size) / stride) + 1
