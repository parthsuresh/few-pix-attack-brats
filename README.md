
# Limited Pixel Attack BraTS
- An extremely limited scenario where the attacker can only perturb a certain number of pixels in the image.
- Semi-black box attack : requires only the probability labels.
- Differential Evolution is used to compute the locations and values of the pixels to be perturbed.

## Objectives
- Determine the effectiveness of limited pixel attacks on brain MRI classification tasks.
- Parallelise differential evolution to make the experiment less computationally taxing.

## Introduction
Neural networks have shown remarkable performance in computer vision tasks like image classification, segmentation and object detection. The use of neural networks in the field of healthcare has been increasing, with neural networks even beating human performance for tasks like chest X-ray analysis. However, the focus of these models is mostly on their performance aspect, and not their reliability. An attacker with malicious intentions can make use of the vulnerabilities of a machine learning model to cause serious harm to a patient's health. For instance, they could attack a model to misclassify one disease as another disease, or not detect a disease at all.

One such attack is the limited pixel attack. The limited pixel attack involves modifying the values of a certain number of pixels in the image, usually much less than the total number of pixels. The pixels to be attacked, as well as the final pixel values after the attack are chosen using the differential evolution algorithm. The differential evolution algorithm searches the entire space of attacks in an efficient manner, which can be parallelized.

In this project, we create a neural network model to classify different types of brain tumors. The BraTS 2018 dataset is used, which has two types of brain tumors, High Grade Glioblastoma(HGG) and Low Grade Glioblastoma(LGG). We then check if our model is susceptible to limited pixel attacks. In order to speed things up, we parallelize the limited pixel attack.

## Model
The model trained is a convolutional neural network, with a structure similar to VGG-16 but with batch-norm layers. We find that our model has an accuracy of 74% when classifying HGG and LGG tumors.

## Differential Evolution
<img width=300 src="./images/de.png">

Each solution has a fixed number of perturbations, and each perturbation is a tuple holding four values : the x-y-z coordinates and the voxel value.

## Results
In progress
