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
[image2]: ./examples/CrossCenter.jpg "Cross Image"
[image3]: ./examples/model.png "Model Architecture Image"
[image4]: ./examples/center.jpg "Center Image"
[image5]: ./examples/center_track2.jpg "Right Image"
[image6]: ./examples/center_right.jpg "Center Right Image"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* model.ipynb containing the script with jupyter to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network working on both Track One and Track Two
* writeup_report.md summarizing the results
* video_track1.mp4 demonstrating the agent driving on Track One
* video_track2.mp4 demonstrating the agent driving on Track Two
* model.png for visualizing the final model

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is build based on [NVIDIA Architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). The NVIDIA architecture provides a powerful end-to-end model to learn to steer with raw pixels from a front-facing camera. Since the application of NVIDIA architecture is the same as the Behavioral Cloning project. I think the NVIDIA architecture is a good base model. 

Based on my training data, I increased the depths of each convolution layer to capture more features in the image. The model consists of a convolution neural network with 5x5 filter sizes, 2x2 stride and depths between 36 and 72 (model.py lines 216-226) for the first three layers. Then a non-stride convolution with 3x3 filter sizes and depth 100 (model.py lines 228-234) in the final two convolution layers. The three fully connected layers are followed the convolutional layers, and lead to a final output control value for the steering angle.

The model includes RELU layers to introduce nonlinearity (model.py lines 217), the image is cropped in the model using Cropping2D to only see the section with road (model.py line 211), and the data is normalized in the model using a Keras lambda layer (model.py line 214). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 218, 222, 226, 230, 234, 238, 242, 246 and 250). 

Also, Weight Regularizer with L2 regularization were implemented for each convolution layer and fully connected layer to reduce overfitting (model.py lines 217, 221, 225, 229, 233, 237, 241, 245 and 249).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 259 - 275). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 258).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road for Track One and driving on the right side of the road for Track Two. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA Architecture. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I created more training data with 2 laps of Tracks One in clockwise direction and 2 laps of Tracks One in counter-clockwise direction.

Next step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially on spots without fence on the right.

![alt text][image1]

To improve the driving behavior in these cases, I added Dropout for each convolutional layer and each fully connected layer. The vehicle is able to drive autonomously around the track without leaving the road for Track One.

Then I created more training data with 2 laps of Track Two in clockwise direction and 2 laps of Track Two in counter-clockwise direction. The car tried to keep on the right side of the road all the time for Track Two. Since the unsymmetrical image captured for Track Two, the augmentation with flip is inapplicable.

Finally, I increased the depths of the convolutional layers and made the vehicle is able to drive autonomously around the both track without leaving the road. However, the vehicle crosses the center line at the spots as shown below.

![alt text][image2]

I think there are ways to improve the model

1. It is a 180 degree right turn, which is difficult for creating training data also. More training data could could be collected for this spot.
2. The shadow may also affect the performance. Augmentation with different brightness and artificial shadow described in [this blog](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) should help. However, I didn't implement this due to the limited time.

####2. Final Model Architecture

The final model architecture (model.py lines 206-253) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         	|     Description	        										| 
|:-----------------:|:-----------------------------------------------------------------:| 
| Input         	| 3@160x320 color image												| 
| Trim image	   	| Crop the image to only see section with road, outputs 3@85x320	|
| Normalization   	| Normalize the image with lambda layer								|  
| Convolution 5x5   | 2x2 stride, valid padding, relu, outputs 36@41x158 				|
| Dropout			| drop probability = 0.2											|
| Convolution 5x5   | 2x2 stride, valid padding, relu, outputs 48@19x77 				|
| Dropout			| drop probability = 0.2											|
| Convolution 5x5   | 2x2 stride, valid padding, relu, outputs 72@8x37					|
| Dropout			| drop probability = 0.2											|
| Convolution 5x5   | 2x2 stride, valid padding, relu, outputs 100@6x35					|
| Dropout			| drop probability = 0.2											|
| Convolution 5x5   | 2x2 stride, valid padding, relu, outputs 100@4x33					|
| Dropout			| drop probability = 0.2											|
| Fully connected 0	| nodes = 4x33x100 = 13200											|
| Dropout			| drop probability = 0.2											|
| Fully connected 1	| nodes = 100														|
| Dropout			| drop probability = 0.2											|
| Fully connected 2	| nodes = 50														|
| Dropout			| drop probability = 0.2											|
| Fully connected 3	| nodes = 10  														|
| Dropout			| drop probability = 0.2											|
| Output			| nodes = 1  														|

The architecture is shown below.

![alt text][image3]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on Track One using center lane driving and four laps on Track Two using right side lane driving. Here is an example image of center lane driving:

![alt text][image4]

Here is an example image of right side lane driving:

![alt text][image5]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The epoch is set to be 20 with early stopping at no improvement for 3 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.