"""This module contains metrics and functions making it easier to work with metrics
in the context of training and testing AI models.
"""

from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from .types import ListOrNpArrayOfNumbers

acc_precision_recall_metrics = [
    accuracy_score, precision_score, recall_score
]
acc_precision_recall_metrics.__doc__ = """A list of default accuracy, precision and recall scores
taken directly from the `sklearn` package.
"""


def dict_average(dict_of_metrics: Dict[str, ListOrNpArrayOfNumbers]) -> Dict[str, np.ndarray]:
    """Averages out all the metrics provided in the form of a dictionary
    mapping `"metric_name": [1, 2, 3]`, where `[1, 2, 3]` is the list
    of a given metric values. For example the input: `"accuracy": [0.1, 0.2, 0.3]` should
    return `0.2`.

    Args:
        dict_of_metrics (Dict[str, ListOrNpArrayOfNumbers]): dictionary mapping names of
                                                             metrics to listed values of
                                                             the given metric

    Returns:
        Dict[str, np.ndarray]: a dictionary of averaged-out metrics
    """
    avg_dict = dict()
    # pylint: disable=invalid-name
    for k, v in dict_of_metrics.items():
        avg_acc = np.mean(v)
        avg_dict[k] = avg_acc
    return avg_dict
