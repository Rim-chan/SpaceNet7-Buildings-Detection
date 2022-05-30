======
LiveEO ML intern challenge
======

**Content**

- `Introduction`_
- `Experimental Setup`_
- `UNet Model`_
- `Scripts`_
- `Instructions for Model Development`_


Introduction
------------
This project studied building footprints detection in satellite imagery, a baseline need for many organizations that can enable many types of analyses. It can be used as a prxoy for population statistics, informe disasters and crises (i.e, disaster and response planning and preparedness: what infrastructure might be at risk), understanding buildings change over time (i.e,rate of urbanization, changes in green spaces). From a humanitarian perspective, such information has a significant impact on preparation, mitigation, and triaging responses and resource allocations to accelerate recovery.

The goal of this Notebook is to develop a neural network based solution to detect buildings in the images provided to accelarate mapping. The analysis is based on a curated subset of the `SpaceNet7 <https://spacenet.ai/sn7-challenge/>`__ dataset. This dataset is hosted on AWS as a Public Dataset.  

This dataset encapsulates the moderate resolution (4m/pixel) EO data, each with 4 channel data (red, green, blue and near-infrared) and a corresponding ground truth mask. We explore the above problem by extracting semantic maps of buildings in medium resolution EO images. Devising it as a supervised learning problem, the state of the art deep neural network architecture, namely, `UNet <https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28>`__, that is commonly used for the the task of semantic segmentation was designed, implemented and experimentally evaluated. 

The main tasks performed are:

- Preprocess/transform data to the format that can be consumed by the model;
- Implement UNet to detect building footprints;
- Train and evaluate the model's results


Experimental Setup
------------------
The experiments carried out in this project are built on top of `pytorch <https://pytorch.org/>`__, an open source machine learning framework in python, and
`MONAI <https://monai.io/>`__ framework - a PyTorch-based open-source foundation for deep learning in healthcare imaging.The resources provided on the `kaggle <https://www.kaggle.com/>`__ platform were used to run these experiments. Kaggle provides free access to NVIDIA TESLA P100 GPUs. These can be used with a quota limit of 30 hours per week. These GPUs are useful for training deep learning models, and they take advantage of GPU-accelerated libraries (e.g. TensorFlow, PyTorch, etc)


UNet Model
----------

UNet consists of an 'encoding' and a 'decoding' part. The encoder is an alternating series of convolution-pooling layers, that extract features from the input, very much like an ordinary classifier. The decoder produces a segmentation map, based on the features derived in the encoder, by alternating transposed convolution layers (or upsampling) and convolution layers. UNet introduces skip-connections between encoder and decoder, at levels where the feature maps have the same lateral extent (number of channels). This enables the decoder to access information from the encoder, such as the general features (edges...) in the original images.
The UNet network depicted in this `paper <https://arxiv.org/pdf/2110.03352.pdf>`__ is the one we used in our project. The source code for this network implemented using MONAI is provided `here <https://docs.monai.io/en/stable/_modules/monai/networks/nets/dynunet.html>`__ . I have also implemented UNet from scratch using plain pytorch (provide below). The MONAI implementation outperformed the the later. Therefore, I decied to use the MONAI UNet. The U-Net that we are using comprises 5 levels. At each stage two convolution operations are applied, each followed by an `Instance normalization <https://paperswithcode.com/method/instance-normalization>`__  and the  `leaky ReLU <https://paperswithcode.com/method/leaky-relu>`__ activation. 

We are using the U-Net model because:
* It is a very simple architecture, which means it is easy to implement and to debug.
* Compared to other architectures, its simplicity makes it faster (less trainable parameters). This is advantageous, as we want to apply the model to a relatively large dataset within a reasonable amount of time to get a first intuition about the data. 
* We can use as much input channels as we want and we are not restricted to only the 3 channels of the standard RGB images. We can also use different sizes of images as in our case (train on (C, 480, 480) image patches and evaluate on the full (C, 1024, 1024)images);
* It has been shown in various cases: `UNet <hhttps://arxiv.org/pdf/1706.06169.pdf>`__ , `DeepUnet <https://https://arxiv.org/pdf/1709.00201.pdf>`__ , `Deep Residual U-Net <https://arxiv.org/pdf/1711.10684.pdf>`__ , that the U-Net is a very capable network with high performance even for satellite imagery. 


Scripts
-------



Instructions for Model Development
----------------

This section provides instructions for the model development phase.

**Download SpaceNet7 data**

This project assumes that you already have the satellite images and their corresponding masks. For privacy reasons the data used here are not public.
The original data is hosted at:

``s3://spacenet-dataset/spacenet/SN7_buildings/``
  
 
**Prepare environment**

.. code:: python
  # install MONAI 
  pip install monai  


.. code:: python

  # import the necessary libraries
  import torch
  import matplotlib.pyplot as plt
  import numpy as np

.. code:: python

  # git clone source
  !git clone https://Rim-chan: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@github.com/Rim-chan/LiveEO.git


**Train segmentation model**

.. code:: python

  !python ./LiveEO/main.py --base_dir "../input/liveeo/LiveEO_ML_intern_challenge" --num_epochs 10 --exec_mode 'train'

**Test segmentation model**

.. code:: python

  !python ./LiveEO/main.py --base_dir "../input/liveeo/LiveEO_ML_intern_challenge" --exec_mode 'evaluate' --ckpt_path './last.ckpt'



**Load and display some samples**

.. code:: python

  preds = np.load('./predictions.npy')   #(6, 1, 1024, 1024)
  lbls = np.load('./labels.npy')         #(6, 1, 1024, 1024)

  # plot some examples
  fig, ax = plt.subplots(1,2, figsize = (20,10)) 
  ax[0].imshow(preds[3][0], cmap='gray') 
  ax[1].imshow(lbls[3][0], cmap='gray') 
