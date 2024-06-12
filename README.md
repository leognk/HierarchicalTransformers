# Hierarchical Transformers

This repository contains all the code used in the study of hierarchical transformers for my Master's thesis at Professor Okatani's Computer Vision Lab at Tohoku University.

## Topics

### New Hierarchical Transformer with Iteratively Refined Hierarchical RAM

The ambitious original motivation behind this project was to design a new kind of architecture that can generate very large content with *high consistency*, such as books or movies. While current models can computationally process large contexts, they cannot generate consistent, high-quality large content. Our proposed architecture is a new kind of hierarchical transformer that defines a hierarchical RAM-like memory that gets iteratively refined. It involves defining a forward and backward pass that iteratively updates tokens at each resolution (the hierarchical RAM) with cross-attention, transmitting information from adjacent resolution levels, and progressively improving the consistency of the representations at each level. Specifically, tokens at resolution level n query tokens at resolution level n - 1 (forward pass) or n + 1 (backward pass) to update their own representation with cross-attention. By doing so, after several updates, the generated content possesses high consistency at any scale.

Unfortunately, in the preliminary experiments, this architecture did not demonstrate significant value in image classification on CIFAR-10/100 and ImageNet-1K and was subsequently abandoned. However, it could show much more value on datasets with very large images and dense prediction tasks, and for generating large content as originally intended. The model is general, making it applicable to any modality such as video, audio, and text. The code can be found in models/encoders/sft/sft.py.

### Hierarchical Transformer on Masked Autoencoding

The project's focus then shifted towards designing an efficient hierarchical transformer for Masked Autoencoding self-supervised pretraining, located at models/ssl/. However, this direction was also unsuccessful, partly due to the high hardware requirements for self-supervised pretraining from scratch, making it difficult to iterate over the experiments.

### Study of Downsampling Layers

Ultimately, I pivoted the project to investigate the importance of the choice of downsampling layer in hierarchical transformers within the context of image classification on ImageNet-1K. The relevant code is in models/encoders/swin.py and models/encoders/metaformer.py. I designed new downsampling layers, namely Attention0Pooling and GeMePooling, which can be found in metaformer.py.

## Directory Structure

The training code is designed for easy customization through various configuration files, offering high modularity. This allows for separate configuration files for datasets, data processing, models, optimizers, training, etc. These config files are located in configs/.

Training is initiated using main_train_cls.py for supervised classification and main_train_ssl.py for self-supervised pretraining.