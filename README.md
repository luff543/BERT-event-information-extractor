# BERT-event-information-extractor

This document describes how to install code on Ubuntu 16.04/18.04.

# Download Anaconda and Install

Go to **[here](https://www.anaconda.com/products/distribution)** and download **Anaconda3-2022.05-Linux-x86_64.sh**

## Requirement install

    $ source /home/<user>/anaconda3/etc/profile.d/conda.sh
    
    $ conda create --name bret_event_information_extractor_env python=3.7
    $ conda activate bret_event_information_extractor_env
    
    $ conda install cudatoolkit=10.0.130 -y
    $ conda install cudnn=7.6.4 -y
    $ conda install tensorflow-gpu=1.14.0 -y
    $ conda install tensorflow=1.14.0 -y
    $ conda install tensorflow-hub=0.8.0 -y
    $ conda install pandas=1.3.2 -y
    $ pip install scikit-learn==0.24.2
    $ pip install pyzmq==19.0.1

# Code

### Train and evaluation model

* Train **Multi-task word-level event field extraction model**

  ``` 
  python train.py