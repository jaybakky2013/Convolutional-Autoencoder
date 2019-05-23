# Convolutional-Autoencoder


The goal of unsupervised learning is to extract important features from unlabelled data, detect and remove redundancies in order to preserve the essential parts of the data. For the past few years,neural network has gained popularity as a machine learning algorithm due to it success in various fields. Convolutional Neural Network(CNN); a type of feed forward neural network which uses convolution in place of general matrix multiplication in at least one of their layers has shown
amazing ability in extracting features of images.

AutoEncoder(AE) is a feedforward neural network that learns efficient data codings in an unsupervised manner. It aims to transform the input to output with least distortion[2]. AE work by compressing the input into a latent-space representation and then reconstructing the output from this representation. It has been applied in different application areas such as dimensionality reduction, reconstructing corrupted data, compact representations of images, Latent space clustering and generative models learning.

One of the advantages of AE over other dimensionality reduction technique is it ability to learn non linear dimension functions. A deep AE extract hierarchical features by its hidden layers; thereby improving the quality of solving specific task. One of the variations of a deep AE [3] is a deep Convolutional Autoencoder (CAE) which, instead of fully-connected layers, contains convolutional layers in the encoder part and deconvolution layers in the decoder part.

In this work we aim to train a Simple autoencoder and Convolutional autoencoder on IMAGENET Dataset for image compression, and compare their performance.
