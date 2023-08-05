"""This module contains wrappers and helpers for the sequence classification task
using Huggingface Transformers package and PyTorch under the hood.

Most of the functionality implemented here I have written on the occasion of:
- MSCI 700 machine learning course at UWaterloo
"""

from typing import List, Iterable, Callable, Tuple, Dict
from pprint import pprint

import transformers
import evaluate
import torch
import numpy as np

from datasets import Dataset
from tokenizers import Tokenizer
from transformers import TrainingArguments, Trainer, EvalPrediction
from torch import nn
from torch.optim import AdamW

from ..pytorch import train
from ..types import Metric
from ..metrics import dict_average


def test_a_batch(model: nn.Module,
                 test_batch: torch.Tensor,
                 attention_mask: torch.Tensor,
                 output_hidden_states: bool) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Runs transformer evaluation on a batched input

    Args:
        model (nn.Module): Huggingface transformer
        test_batch (torch.Tensor): batched `input_ids`
        attention_mask (torch.Tensor): attention mask
        output_hidden_states (bool): whether to return also the attention weights

    Returns:
        torch.Tensor | (torch.Tensor, torch.Tensor): output logits and optionally
                                                     the attention weights
    """
    with torch.no_grad():
        torch_outs = model(
            test_batch,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
        )
    return torch_outs


def test_and_calculate_metrics(tokens_ids: torch.Tensor,
                               labels: torch.Tensor,
                               model: nn.Module,
                               metrics: List[Metric] | Metric,
                               split_into: int = 4,
                               output_hidden_states: bool = False) -> Dict[str, List[int | float]]:
    """Runs transformer model evaluation and produces a metric dict

    Args:
        tokens_ids (torch.Tensor): tokenized inputs
        labels (torch.Tensor): labels for sequence classification task
        model (nn.Module): transformer model
        metrics (List[Metric] | Metric): metrics to be calculated
        split_into (int, optional): sub-batching factor to save VRAM. Defaults to 4.
        output_hidden_states (bool, optional): whether to return attention maps. Defaults to False.

    Returns:
        Dict[str, List[int | float]]: dict of metrics calculated for each batch
    """
    metric_vals = dict()

    if not isinstance(metrics, list):
        # wrap singular metric into a list to use the same interface downstream
        metrics = [metrics]

    for metric in metrics:
        # initialize lists where we will store collected metrics
        metric_vals[metric.__name__] = []

    slice_size = tokens_ids.shape[0] // split_into
    for test_batch, batch_labels in zip(tokens_ids.split(slice_size),
                                        torch.tensor(labels).split(slice_size)): # pylint: disable=no-member

        # Compute the embeddings:
        # attention_mask = test_batch != tokenizer.pad_token_id
        attention_mask = torch.ones_like(test_batch) # pylint: disable=no-member

        # Send tokens and attention mask to the GPU:
        test_batch = test_batch.to("cuda")
        # attention_mask = attention_mask.to("cuda")

        # Model outputs:
        torch_outs = test_a_batch(model,
                                  test_batch,
                                  attention_mask,
                                  output_hidden_states)

        y_hat_prob = nn.Sigmoid()(torch_outs.logits)
        y_hat = torch.argmax(y_hat_prob, axis=-1) # pylint: disable=no-member
        for metric in metrics:
            metric_value = metric(batch_labels.to("cpu").detach().numpy(), y_hat.to("cpu").detach().numpy())
            metric_vals[metric.__name__].append(metric_value)

    return metric_vals



def accuracy_metric_trainer_api(eval_pred: EvalPrediction) -> Dict[str, int | float]:
    """Calculates accuracy metric using the `evaluate` package. Compatible
    with Huggingface Trainer API.

    Args:
        eval_pred (EvalPrediction): a prediction instance from the Huggingface model

    Returns:
        Dict[str, int | float]: calculated accuracy
    """
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_pred
    pred_class = np.argmax(logits, axis=-1)  # take the max-scoring logit as the predicted class ID
    return accuracy.compute(predictions=pred_class,
                            references=labels)


def tokenize_default(tokenizer: Tokenizer) -> Iterable[int]:
    """Default tokenize function for the `.map()` interface in Huggingface Transformers.

    Args:
        tokenizer (Tokenizer): a tokenizer instance to be used

    Returns:
        Callable[[Dataset], List[int]]: the concrete tokenization function
    """
    def tokenize(dataset: Dataset):
        return tokenizer(dataset["sequence"], padding=True)
    return tokenize


TokenizerClosure = Callable[[Tokenizer], Callable[[Dataset], Iterable[int]]]
TokenizerClosure.__doc__ = "The interface of the function `tokenize_default`"


# pylint: disable-next=too-many-arguments
def train_using_trainer_api(tokenizer,
                            model,
                            training_args,
                            train_dataset,
                            val_dataset,
                            tokenize_closure: TokenizerClosure = tokenize_default,
                            batch_tokenize: bool = True):

    """Uses the Trainer API from Huggingface Transformers for fine-tuning.
    This wrapper accounts for basic metrics collection as well.

    Returns:
        transformers.Trainer: the Trainer instance which can be started via
                              `.train()` method
    """

    tokenize = tokenize_closure(tokenizer)
    tokenized_train_dataset = train_dataset.map(tokenize, batched=batch_tokenize)
    tokenized_val_dataset = val_dataset.map(tokenize, batched=batch_tokenize)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=accuracy_metric_trainer_api
    )

    return trainer


# pylint: disable-next=too-many-arguments
def test(tokenizer: Tokenizer,
         model: nn.Module,
         test_dataset: Dataset,
         metrics: List[Metric],
         sequence_property_name: str = "sequence",
         label_property_name: str = "label",
         split_into: int = 4):
    """Runs evaluation over a complete test dataset (`Dataset` instance).

    Args:
        tokenizer (Tokenizer): tokenizer instance
        model (nn.Module): model instance
        test_dataset (Dataset): test dataset
        metrics (List[Metric]): metrics to compute
        sequence_property_name (str, optional): which property in the dataset contains
                                                the sequence to be fed to the transformer.
                                                Defaults to "sequence".
        label_property_name (str, optional): which property in the dataset contains labels
                                             for sequence classification. Defaults to "label".
        split_into (int, optional): sub-batching factor to conserve VRAM. Defaults to 4.
    """

    tokens_ids = tokenizer(test_dataset[sequence_property_name], return_tensors="pt")["input_ids"]

    metric_vals = test_and_calculate_metrics(tokens_ids,
                                             test_dataset[label_property_name],
                                             model,
                                             metrics,
                                             split_into)

    avgs_of_metric_vals = dict_average(metric_vals)

    pprint(f"Metrics averaged over batches: {avgs_of_metric_vals}", indent=4)


ModelLoader = Callable[[], nn.Module]


# pylint: disable-next=too-many-arguments
def train_and_test(tokenizer: Tokenizer,
                   model: nn.Module,
                   training_args: TrainingArguments,
                   train_dataset: Dataset,
                   val_dataset: Dataset,
                   test_dataset: Dataset,
                   metrics: List[Metric],
                   seed: int = 42,
                   device: str = "cuda"):
    """Collects the training and testing pipelines for a transformer model
    into one function.

    Args:
        tokenizer (Tokenizer): tokenizer
        model (nn.Module): model
        training_args (TrainingArguments): training arguments
        train_dataset (Dataset): training dataset
        val_dataset (Dataset): validation dataset
        test_dataset (Dataset): testing dataset
        metrics (List[Metric]): list of metrics to be collected
        seed (int, optional): random seed used for initialization. Defaults to 42.
        device (str, optional): which device to use. Defaults to "cuda".
    """
    transformers.set_seed(seed)
    # If you enjoy faster training times use CUDA
    model.to(device)
    # Train:
    trainer = train_using_trainer_api(tokenizer, model, training_args, train_dataset, val_dataset)
    trainer.train()
    # Test:
    test(tokenizer, model, test_dataset, metrics)


# pylint: disable-next=too-many-arguments
def monte_carlo_train_and_test_pipeline(tokenizer: Tokenizer,
                                        model_loader: ModelLoader,
                                        training_args: TrainingArguments,
                                        train_dataset: Dataset,
                                        val_dataset: Dataset,
                                        test_dataset: Dataset,
                                        metrics: List[Metric],
                                        runs: int = 3,
                                        seed: int = 42,
                                        device: str = "cuda"):
    """Applies a Monte-Carlo strategy to model training and testing.
    The model will be re-loaded at each iteration and will be initialized
    using a different random seed. The `seed` argument in the function signature
    controls the seeds which will be used for model initialization.

    Args:
        tokenizer (Tokenizer): the tokenizer object
        model_loader (Callable[[], nn.Module]): a lambda function returning the model
        training_args (TrainingArguments): training arguments
        train_dataset (Dataset): training dataset
        val_dataset (Dataset): validation dataset
        test_dataset (Dataset): test dataset
        metrics (List[Metric]): a list of metrics to be calculated
        runs (int, optional): how many runs to perform. Defaults to 3.
        seed (int, optional): random seed. Defaults to 42.
    """
    np.random.seed(seed)
    seeds = np.random.random_integers(1, 100, runs)
    print(f"Starting Monte Carlo Runs with seeds: {seeds}")

    for secondary_seed in seeds:
        print(f"Using seed: {secondary_seed}")
        train_and_test(tokenizer,
                       model_loader(),
                       training_args,
                       train_dataset,
                       val_dataset,
                       test_dataset,
                       metrics,
                       secondary_seed,
                       device)


def train_from_scratch(tokenized_train_dataset: Iterable[List[int], int],
                       model: nn.Module,
                       epochs: int = 3,
                       device: str = "cuda"):
    """A from-scratch training loop function for a sequence-classification
    transformer.

    Args:
        tokenized_train_dataset (Iterable[List[int], int]): tokenized inputs with
                                                            labels passes as tuples
                                                            of `(token_seq, label)`
        model (nn.Module): transformer model
        epochs (int, optional): number of epochs. Defaults to 3.
        device (str, optional): device to train on. Defaults to "cuda".
    """
    # Set up the optimizer
    optimizer = AdamW(model.parameters())

    # pylint: disable=no-member
    input_ids = torch.tensor([i[0] for i in tokenized_train_dataset])
    input_ids = input_ids.to(device)
    # pylint: disable=no-member
    labels = torch.tensor(i[1] for i in tokenized_train_dataset)
    labels = labels.to(device)

    train(input_ids,
          labels,
          model,
          optimizer,
          None,
          epochs,
          device,
          model_kwargs=dict(labels=labels))
