# Hierarchical Transformers

This repository contains all the code used in the study of hierarchical transformers for my master's thesis.

## Topic

Originally, this project aimed to design a new kind of hierarchical transformer for efficiently processing very large contexts, such as whole-slide images. This involved defining a forward and backward pass that iteratively updates tokens at each resolution, transmitting information from adjacent resolution levels. Unfortunately, this architecture did not demonstrate significant value in image classification tasks on CIFAR-10/100 and ImageNet-1K and was subsequently abandoned. The code for this can be found in models/encoders/sft/sft.py.

The project's focus then shifted towards designing an efficient hierarchical transformer for Masked Autoencoding self-supervised pretraining, located at models/ssl/. However, this direction was also unsuccessful, partly due to the high hardware requirements for self-supervised pretraining from scratch.

Ultimately, I pivoted the project to investigate the importance of the choice of downsampling layer in hierarchical transformers within the context of image classification on ImageNet-1K. The relevant code is in models/encoders/swin.py and models/encoders/metaformer.py. I designed new downsampling layers, namely Attention0Pooling and GeMePooling, which can be found in metaformer.py.

## Directory Structure

The training code is designed for easy customization through various configuration files, offering high modularity. This allows for separate configuration files for datasets, data processing, models, optimizers, training, etc. These config files are located in configs/.

Training is initiated using main_train_cls.py for supervised classification and main_train_ssl.py for self-supervised pretraining.