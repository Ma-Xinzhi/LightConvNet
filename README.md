# LightConvNet
## A Temporal Dependency Learning CNN with Attention Mechanism for MI-EEG Decoding [[paper]](https://doi.org/10.1109/TNSRE.2023.3299355)
This is the PyTorch implementation of the temporal dependency learning CNN for MI-EEG decoding.
## Network Architecture
![Network architecture](https://github.com/Ma-Xinzhi/LightConvNet/blob/main/network_architecture.png)
The temporal dependency learning CNN is designed with the aim of learning temporal dependencies between discriminative features in different time periods during MI tasks. It is composed of the following five stages:
1. Multi-view data representation: The multi-view representation of the EEG data is obtained by spectrally filtering the raw EEG with multiple narrow-band filters.
2. Spatial and spectral information learning: The spatial and spectral discriminative patterns are then learned using a spatial convolution block.
3. Temporal segmentation and feature extraction: A series of non-overlapped time windows is employed to segment the output data, and the discriminative feature is further extracted from each time window to capture MI-related patterns generated in different stages.
4. Temporal attention module: A novel temporal attention module is designed to explore temporal dependencies between discriminative features in different time windows.
5. Classification: A fully connected (FC) layer finally classifies features from the temporal attention module layer into given classes.
## Requirements
* PyTorch 1.7
* Python 3.7
* mne 0.23
## Datasets
* [BCI_competition_IV2a](https://www.bbci.de/competition/iv/)
* [OpenBMI](http://gigadb.org/dataset/view/id/100542)
## Toolbox
This repository is designed as a toolbox that provides all necessary tools for training and testing the proposed network. All the data functionalities are defined in the data directory. After preprocessing data, the cv.py and train_test.py are the entry points to train and test the proposed network in the session-dependent setting and session-independent setting (defined in the paper), respectively.
## Results
The classification results for our proposed network and other competing architectures are as follows:
![Results](https://github.com/Ma-Xinzhi/LightConvNet/blob/main/results.png)
