# Distracted-Driver-Detection
The goal is to detect and classify different distractions of driver using deep learning.

# Dataset
The dataset is obtained from https://www.kaggle.com/c/state-farm-distracted-driver-detection/data
The dataset contains 22,224 images belonging to one of the 10 classes given below:
•	c0: safe driving
•	c1: texting - right
•	c2: talking on the phone - right
•	c3: texting - left
•	c4: talking on the phone - left
•	c5: operating the radio
•	c6: drinking
•	c7: reaching behind
•	c8: hair and makeup
•	c9: talking to passenger

# Model description
A convolutional neural network is implemented using Keras to achieve the goal.
The first convolution layer uses 32 feature detectors (or filters) having dimension 3x3 that convolves with input volume [64 × 64 × 3 (for R, G, B channels)]. Each feature detector corresponds to a slice or a depth of one in the output volume. Activation function relu is used to have non-linearity in the network. Max-pooling reduces the size of the input volume (leaving the depth unchanged) to prevent the model from overfitting and getting too large (too many weights to compute). Our model includes two more such convolution layers with 32 & 64 feature detectors respectively.
The flatten layer transforms the 3D output of convolution layer into a 1D vector.
The fully connected (FC) layer is like ordinary NN where each neuron is connected to all the outputs from the previous layer. The last FC layer computes probabilities for each class (each neuron corresponds to a class). Softmax trains the final layer to correctly predict with maximal confidence for each image.
Model is then configured for training using compile(). 

# Preprocessing
The dataset is split into training set and validation set consisting 20059 & 2165 images respectively.
An augmented image generator is created for training set using ImageDataGenerator() to boost the performance of the network. Parameters such as rescale, shear_range, zoom_range, horizontal_flip are used.
An image generator for validation set is created to rescale pixel values to range from 0 to 1. 
Data generators take data from specified directory using flow_from_directory() and generate batches of augmented/normalized data.

# Training
The model is trained using fit_generator(). 
