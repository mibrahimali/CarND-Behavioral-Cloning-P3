# Behaviorial Cloning Project

Nowdays the technology focus on creating  Autonomous vehicles. Deep learning takes a major role in such progress in the field.

In this project, End to End Control Pipeline of vehicle Steering Angle is developed using Deep learning methods. The Main purpose is that autonomously drive a vehicle in a Simulator using only Front Facing camera Images


[//]: # (Image References)

[image1]: ./readme_images/normal.png "Normal Sample of center Camera Image"
[image2]: ./readme_images/flipped.png "Flipped Sample of center Camera Image"
[image3]: ./readme_images/center.jpg "Center Camera image with Steering angle = 0.0"
[image4]: ./readme_images/left.jpg "Left Camera image with Steering angle = 0.15"
[image5]: ./readme_images/right.jpg "Right Camera image with Steering angle = -0.15"
[image6]: ./readme_images/model.png "Custom Nvidia Model"
[image7]: ./model_nvidia_bt_512_e_25_resize_60_200_lr_5e-4_s_32256/model_nvidia_bt_512_e_25_resize_60_200_lr_5e-4_s_32256.png "Training Loss"
[image8]: ./readme_images/Training_dataset_Steering_command_Histogram.png "Training dataset Steering command Histogram"
[image9]: ./readme_images/Validation_dataset_Steering_command_Histogram.png "Validation dataset Steering command Histogram"
[image10]: ./readme_images/auto.gif "Testing Model"

## Overview
---
This repository contains files for the Behavioral Cloning Project as Follow:

* dataset_augmentor : contains script for dataset preperation and augmentation
* dataset_analysis : contains script for dataset analysis for ex. histogram of dataset
* dataset_generator : contains dataset generator function and set of augmentatoin steps on data
* train.py : contains the script to create and train the model
* train_log.csv : contains Training dataset 
* Valid_log.csv : contains Validation dataset
* drive.py : for driving the car in autonomous mode
* model_nvidia_bt_512_e_25_resize_60_200_lr_5e-4_s_32256 folder:
  * model_nvidia_bt_512_e_25_resize_60_200_lr_5e-4_s_32256.h5: contains a trained convolution neural network 
  * model_nvidia_bt_512_e_25_resize_60_200_lr_5e-4_s_32256.mp4: contains a screen recorded video of a full laps on track 1.
* README.md summarizing the results


## Data Set Preparation:
### Creation of the Training Set

1. Using Simulator Provided by Udacity, I first recorded two laps on track one in the forward direction along with dataset sample provided by Udaicty also To capture good driving behavior.
2. Also record two laps in reverse dirction for further generalization of dataset
3. to increase generalization of dataset also rondom image on training phase is flipperd horizontally as shown here
![alt text][image1] ![alt text][image2]

```python
import numpy as np
Flipped_Image = np.fliplr(Image)
Flipped_Steering_Angle = -1 * float(steering_angle)
```
4. for further increase my Training data, images from both left and right shifted front facing cameras were used to simulate bad examples to the network instead of driving recovery laps in the simulator
![alt text][image4] ![alt text][image3] ![alt text][image5]

a manully tunned steering correction factor is add to these images = 0.15 as follow

```python
Left_steering = center_steering + correction value
Right_steering = center_Steering - correction value
```
5- most of data set is baised to zero steering angle so filtering process was done to normlize dataset 

__Final Dataset for both Training and Validation__
![alt text][image8]
![alt text][image9] 

* Training Dataset size = 32544
* Validation Dataset size = 1713

### Design and Test a Model Architecture

__Model Architecture__

Starting with well known Nvidia CNN Architecture for "End to End Learning for self-driving cars" which can be found [in this paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). as first step and modifying this model as follow
* introducesing two Dropout layers to prevent overfitting 
* removing last convolution layer and fully connected layer of 50 neuran to simplify model. 
* Using Elu as Activation function instead of normal relu to enhance learning capabilites

My final model is presented in next figure:

![alt text][image6]

* Total params: 479,541
* Trainable params: 479,541
* Non-trainable params: 0

__Training Phase__ 

After Multiple Tunning cycles of parameters
1. Adam Optimizer Used with learning rate = 5e-4
2. Batch Size = 512 Images
3. Epochs Numbers = 25
* Following Figure show Training loss over epochs

![alt text][image7]

Finally Video of Model Testing can be found in model_nvidia_bt_512_e_25_resize_60_200_lr_5e-4_s_32256 Folder

![alt text][image10]

__Further Improvment__

1. Introduce translation factor on Training Images to enhance stability of model by learning intermadiate recovery steps
2. Using LSTM Network as Steering command normally depend not only on this time stamp but also on history of commands
3. generalize Training Dataset by augmenting lights and shadow effect to generalize netowork for Track 2