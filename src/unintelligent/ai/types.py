"""Type aliases for the `ai` module"""

import torch
import numpy as np
import numpy.typing as npt
from typing import Callable, List

ListOrNpArray = npt.NDArray | List
ListOrNpArrayOfNumbers = npt.NDArray[np.int] | npt.NDArray[np.float_] | List[int] | List[float]
TensorOrNumPyArray = torch.Tensor | np.ndarray
Metric = Callable[[TensorOrNumPyArray, TensorOrNumPyArray], TensorOrNumPyArray]
