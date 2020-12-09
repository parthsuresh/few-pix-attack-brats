
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

An example of limited pixel attacks i.e. the one-pixel attack where only one pixel can be modified is shown below :

<img width=500 src="./images/one-pix.jpg">


In this project, we create a neural network model to classify different types of brain tumors. The BraTS 2018 dataset is used, which has two types of brain tumors, High Grade Glioblastoma(HGG) and Low Grade Glioblastoma(LGG). We then check if our model is susceptible to limited pixel attacks. In order to speed things up, we parallelize the limited pixel attack.


## Model
The model trained is a convolutional neural network, with a structure similar to VGG-16 but with batch-norm layers. We find that our model has an accuracy of 74% when classifying HGG and LGG tumors.

## Differential Evolution
<img width=300 src="./images/de.png">

The perturbations to the image are encoded as an array that is optimized using differential evolution. Each candidate solution has a fixed number of perturbations, and each perturbation is a tuple holding four values : the x-y-z coordinates and the voxel value (voxels are the basic units of 3D images, analogous to pixels for 2D images). Each perturbation modifies one pixel.

Every iteration produces new candidates according to the equation <img src="https://render.githubusercontent.com/render/math?math=x_{i}(g %2B 1) = x_{r1}(g) %2B F(x_{r2}(g) - x_{r3}(g))"> where x is the array of candidates, g is the iteration step and r1, r2 and r3 are not equal. The fitness of the generated candidates are compared to their parents. If the new candidate does better than its parent, then it replaces its parent.

In our case, fitness is the value of the probability label. For instance, if we are attacking an image with an HGG tumor, the probability of the image to be an HGG tumor as predicted by the neural network is the fitness value. So, the candidate with the lower probability would be retained by our algorithm, since our aim is to make the network predict the wrong value. If this probability goes below 50%, we say that the attack is succesful.

## Parallelizing Differential Evolution

We implement two ways to parallelize the code:

- Since attacks on different images are independent of each other, we assign different images to different processes. This brings down the total time by an amount close to the number of processes spawned.
- On speed profiling our code, we notice that the main bottleneck of the process is the evaluation of fitness of the candidates. We extend the first approach to include threads that independently evaluate the fitness of the candidates. So, we have n processes with m threads per process. Each image is assigned to one of the n processes, and the candidates created for an attack on that image are evaluated by the m threads that belong to the process.

## Results

The attack is unsuccesful for all 8 images in our test set (10 candidates and 10 iterations), implying that our model is safe and resistant towards limited pixel attacks.

Parallelizing the code using the first approach mentioned above gives the following runtimes : 
- 4 processes : 2125.37 s
- 2 processes : 4310.76 s
- 1 process : 8701.48s (same as serial execution)

Parallelizing the code using the second approach mentioned above gives the following runtimes : 
- 4 processes , 4 threads per process : 515.928 s
- 4 processes , 2 threads per process : 1019.91 s
- 4 processes - 1 thread per process : 2112.37 s


## Conclusions

- Preliminary tests show that our model is safe towards limited pixel attacks.
- By parallelizing the code, we reduce the processing time from 8701 s to 516 s, i.e. to nearly 1/16th of its original value.

## Future Work

- Increasing the number of candidates and number of iterations will make the testing more rigorous and ensure reduce any further uncertainity about the safety of the model.

