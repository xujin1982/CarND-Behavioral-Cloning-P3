#**Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/NoFence.png "No Fence"
[image2]: ./examples/center.jpg "Center Image"
[image3]: ./examples/center_left.jpg "Center Left Image"
[image4]: ./examples/center_right.jpg "Center Right Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is build based on [NVIDIA Architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

The model consists of a convolution neural network with 5x5 filter sizes, 2x2 stride and depths between 24 and 48 (model.py lines 99-109) for the first three layers. Then a non-stride convolution with 3x3 filter sizes and depth 64 (model.py lines 111-117) in the final two convolution layers. The three fully connected layers are followed the convolutional layers, and lead to a final output control value for the steering angle.

The model includes RELU layers to introduce nonlinearity (model.py lines 100), the data is normalized in the model using a Keras lambda layer (model.py line 93), and the image is cropped in the model using Cropping2D to only see the section with road (model.py line 97). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 101, 105, 109, 113, 117, 121, 125, 129 and 133). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 143 - 147). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 142).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA Architecture. I thought this model might be appropriate because it worked for NVIDIA's self-driving car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I created more training data with 2 laps of Tracks 1 in clockwise direction and 2 laps of Tracks 1 in counter-clockwise direction. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially on spots without fence on the right.

![alt text][image1]

To improve the driving behavior in these cases, I added Dropout for each convolutional layer and each fully connected layer. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 89-136) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         	|     Description	        										| 
|:-----------------:|:-----------------------------------------------------------------:| 
| Input         	| 3@160x320 color image												| 
| Normalization   	| Normalize the image with lambda layer								| 
| Trim image	   	| Crop the image to only see section with road, outputs 3@85x320	| 
| Convolution 5x5   | 2x2 stride, valid padding, relu, outputs 24@41x158 				|
| Dropout			| drop probability = 0.5											|
| Convolution 5x5   | 2x2 stride, valid padding, relu, outputs 36@19x77 				|
| Dropout			| drop probability = 0.5											|
| Convolution 5x5   | 2x2 stride, valid padding, relu, outputs 48@8x37					|
| Dropout			| drop probability = 0.4											|
| Convolution 5x5   | 2x2 stride, valid padding, relu, outputs 64@6x35					|
| Dropout			| drop probability = 0.3											|
| Convolution 5x5   | 2x2 stride, valid padding, relu, outputs 64@4x33					|
| Dropout			| drop probability = 0.3											|
| Fully connected 0	| nodes = 4x33x64 = 8448											|
| Dropout			| drop probability = 0.3											|
| Fully connected 1	| nodes = 100														|
| Dropout			| drop probability = 0.2											|
| Fully connected 2	| nodes = 50														|
| Dropout			| drop probability = 0.2											|
| Fully connected 3	| nodes = 10  														|
| Dropout			| drop probability = 0.2											|
| Output			| nodes = 1  														|


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

To augment the data sat, I also flipped images and angles thinking that this would simply   increase the training data set and have angles for both steering left and right. For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image4]


I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as evidenced by little improvement of both MSE of training data set and validation data set. I used an adam optimizer so that manually training the learning rate wasn't necessary.