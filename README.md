# Dataloader

A python module for loading datasets from 

* Scikit-learn
* Torchvision

It supports loading

* Wine
* Iris
* MNIST datasets

Defaults to loading IRIS.

This module also performs pre-processing relevant to each data set type.
It handles categorical data for tabular sets like Iris and it performs transforms for images in the computer vision dataset like MNIST.

## Install

Using a package manager
```
$ poetry add dataloader-fetcher
```

## Example usage

```
fetcher = DataloaderFetcher()
train_loader = fetcher.train_loader(name="Iris")
test_loader  = fetcher.test_loader(name="Iris")
```