"""Dataloader module"""

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch import is_tensor, cuda
from torchvision import transforms

from sklearn import datasets as skl_datasets  # pylint: disable=C0412
from torchvision import datasets as tv_datasets  # pylint: disable=C0412
from sklearn.model_selection import StratifiedShuffleSplit


# pylint: disable=E1101
DEVICE = torch.device("cuda:0" if cuda.is_available() else "cpu")


class DataLoaderFetcher:
    """DataLoaderFetcher class"""

    def __init__(self, name: str = "Iris"):
        self.name = name

    # pylint: disable=W0511
    # TODO: Handle error in case user passes an unsupported dataset name
    def train_loader(self) -> DataLoader:
        """Returns a DataLoader for the training set"""

        if self.name == "Wine":
            return DataLoader(
                self.dataset(train=True),
                batch_size=40,
                shuffle=True,
                drop_last=False,
            )

        if self.name == "MNIST":
            transform_train = transforms.Compose(
                [
                    # Image Transformations suitable for MNIST dataset(handwritten digits)
                    transforms.RandomRotation(30),
                    transforms.RandomAffine(
                        degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    # Mean and Std deviation values of MNIST dataset
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            return DataLoader(
                self.dataset(train=True, transform=transform_train),
                batch_size=125,
                shuffle=True,
                num_workers=2,
            )

        return DataLoader(
            self.dataset(train=True),
            batch_size=40,
            shuffle=True,
            drop_last=False,
        )  # Iris dataset is default

    def test_loader(self) -> DataLoader:
        """Returns a DataLoader for the test set"""

        if self.name == "Wine":
            return DataLoader(
                self.dataset(train=False),
                batch_size=10,
                shuffle=True,
                drop_last=False,
            )

        if self.name == "MNIST":
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            return DataLoader(
                self.dataset(train=False, transform=transform_test),
                batch_size=100,
                shuffle=False,
                num_workers=2,
            )

        return DataLoader(
            self.dataset(train=False),
            batch_size=10,
            shuffle=True,
            drop_last=False,
        )  # iris dataset is default

    def dataset(self, train=True, transform=None):
        """Returns a dataset"""
        if self.name == "Wine":
            return TabularDataSet(train=train, dataset=skl_datasets.load_wine)
        if self.name == "MNIST":
            return tv_datasets.MNIST(
                root="./data", train=train, download=True, transform=transform
            )
        return TabularDataSet(train=train, dataset=skl_datasets.load_iris)


class TabularDataSet(Dataset):
    """Tabular Dataset class"""

    def __init__(self, train=True, dataset=None):
        if dataset is None:
            raise RuntimeError("Dataset not provided")

        data_bundle = dataset()

        predictors_x, targets_y = data_bundle.data, data_bundle.target
        y_categorical = to_categorical(targets_y)

        self._number_of_predictors = np.shape(predictors_x)[1]
        self._number_of_categories = np.shape(y_categorical)[1]

        stratified_shuffle_split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=123
        )

        for train_index, test_index in stratified_shuffle_split.split(
            X=predictors_x, y=targets_y
        ):
            x_train_array = predictors_x[train_index]
            x_test_array = predictors_x[test_index]
            y_train_array = y_categorical[train_index]
            y_test_array = y_categorical[test_index]

        if train:

            self.data = x_train_array
            self.target = y_train_array
        else:

            self.data = x_test_array
            self.target = y_test_array

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        # pylint: disable=W0511
        # TODO: may be repetition of from_numpy here
        predictors = (
            torch.from_numpy(self.data[idx, :]).float().to(DEVICE)
        )  # pylint: disable=E1101
        categories = torch.from_numpy(self.target[idx]).to(
            DEVICE
        )  # pylint: disable=E1101
        return predictors, categories

    def number_of_predictors(self):
        """Returns the number of predictors"""
        return self._number_of_predictors

    def number_of_categories(self):
        """Returns the number of categories"""
        return self._number_of_categories

    def __len__(self):
        return len(self.data)


# pylint: disable=C0301
# Please refer to Keras: https://github.com/keras-team/keras/blob/14f71177ad28a60a4ea41775b2ac159d3688c792/keras/utils/np_utils.py#L22-L74


def to_categorical(y_class_values, num_classes=None, dtype="float32"):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with `categorical_crossentropy`.

    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
            as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'float32'`.

    Returns:
        A binary matrix representation of the input. The class axis is placed
        last.

    Example:

    >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
    >>> a = tf.constant(a, shape=[4, 4])
    >>> print(a)
    tf.Tensor(
        [[1. 0. 0. 0.]
        [0. 1. 0. 0.]
        [0. 0. 1. 0.]
        [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)

    >>> b = tf.constant([.9, .04, .03, .03,
    ...                  .3, .45, .15, .13,
    ...                  .04, .01, .94, .05,
    ...                  .12, .21, .5, .17],
    ...                 shape=[4, 4])
    >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.82807 0.1011  1.77196]

    >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0. 0.]
    """
    y_class_values = np.array(y_class_values, dtype="int")
    input_shape = y_class_values.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y_class_values = y_class_values.ravel()
    if not num_classes:
        num_classes = np.max(y_class_values) + 1
    num_columns_y = y_class_values.shape[0]
    categorical = np.zeros((num_columns_y, num_classes), dtype=dtype)
    categorical[np.arange(num_columns_y), y_class_values] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
