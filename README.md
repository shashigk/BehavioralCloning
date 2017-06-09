**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/road_center.png 			"Sample Image"
[image2]: ./examples/road_flip.png   			"Flipped version of the Sample Image"
[image3]: ./examples/road_left.png  			"Sample Image From Left Camera"
[image4]: ./examples/road_right.png 			"Sample Image From Right Camera"
[image5]: ./examples/orig_data_hist.png  		"Original Data Distribution"
[image6]: ./examples/augm_data_hist.png  		"Augmented Data Distribution"
[image7]: ./examples/center_2017_04_15_22_46_59_452.png "Close to Edge"
[image8]: ./examples/center_2017_04_15_22_47_05_906.png "Begin Recovery"
[image9]: ./examples/center_2017_04_15_22_47_06_136.png "Continue Recovery"
---
####1. Project Files

My project includes the following files:
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode.
* **model.h5** containing a trained convolution neural network 
* **writeup\_report.md** summarizing the results.
* **BehClo.ipynb** Notebook to generate sample images and histograms.
* **d100.mp4** Video recording of a sucessful run.

I also tried a second model, where I had a smaller Fully Connected layer, 
immediately after the flattening the convolutions, and it gave slightly smoother
The files for the second model are along with a sample video are below.
* **model\_d50.py** Code for the second model, which just has a smaller dense layer after flattening layer.
* **model\_d50.h5** Model file for the second model.
* **d50.mp4** A smoother video obtained by seond different model.

####2. Using submitted code.
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model file for the second model is model\_d50.h5 and can be used similarly.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I use a *variant* NVIDIA model, presented in the course along with the following modifications:
* Normalization to have intensity values between -0.5 to +0.5.
* Cropping to consider only the relevant subset of the image.
* Dropout of 0.6 to prevent overfitting.
The training architecuture is in the procedure 'trainNvidiaModel' in the file model.py.


####2. Attempts to reduce overfitting in the model

The dropout layer in the function 'trainNvidiaModel' helps reduce overfitting.
I tried dropouts of 0.5, 0.6. 0.7 and the model with dropout value of 0.6 
was able to drive around the track successfully.

####3. Model parameter tuning

Since this was a regression model, I used mean square error as the metric to
be minimized. The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

I drove a few laps around the left track. During this process while I was mostly
focussed on center lane driving, there were also certain situations where I
tried to recover from being close to the edge of the road.

For example, in the image below the car is close to the right edge of 
the road

![Close to Edge][image7]


The recovery then begins as shown in the image below.

![Begin Recovery][image8]

The image below shows the car has moved back closer to the center.

![Continue Recovery][image9]


This was important for the regression model to learn the steering angles
in order to navigate around the curves.

###Model Architecture and Training Strategy

####1. Solution Design Approach

Since the NVIDIA model has shown to be reasonable model to work with the simulator data,
I decided to use it as a starting point. However, I added normalization, cropping and drop out
as described below.

The architecture is summarized below.

| Layer         	|     Description	        		          | 
|:---------------------:|:-------------------------------------------------------:| 
| Input         	| 160x320x3 RGB image				          | 
| Normalization         | Ensure values between \[-0.5, +0.5]		          |
| Cropping              | Remove top 70 and bottom 25 rows, output 65x320x3       |
| Convolution      	| 5x5 Kernel, 2x2 strides, 24 dim output, relu activation |
| Convolution      	| 5x5 Kernel, 2x2 strides, 36 dim output, relu activation |
| Convolution      	| 5x5 Kernel, 2x2 strides, 48 dim output, relu activation |
| Convolution      	| 3x3 Kernel, 64 dim output, relu activation              |
| Convolution      	| 3x3 Kernel, 64 dim output, relu activation              |
| Flatten 		| Flatten to 2112 layers			          |
| Fully connected	| Output 100	 	                                  |
| Dropout               | Prob = 0.6					          |
| Fully Connected       | Output 50						  |
| Dropout               | Prob = 0.6					          |
| Fully Connected       | Output 10						  |
| Fully Connected       | Output 1						  |


*Note*: For the second model, the Fully Connected layer after flattening
convolution had output size of 50.

The data gathered was split into training and validation set.
I trained the data for 5 epochs and training loss continued to decrease.
The validation loss decreased initially, but then started increase.
Since this could be potentially due to overfitting, I introduced dropout layers.


####2. Creation of the Training Set & Training Process

I recorded several laps through track 1. During the recording I mostly kept the
vehicle to the center as shown below. But there were situations where I
had to also recover from the sides of the road. A sample image recorded on
the center camera is shown below.

![Center Driving][image1]

I found that the steering angle distribution was skewed as shown in
the histogram below. 
![Original data histogram][image5]

So I decided to augment the data set as follows:
Use images from the left and the right cameras

For the sample image shown above the image from the left camera is below.

![Left Camera][image3]


The right camera image is shown below.

![Right Camera][image4]


The steering angle for the left image (resp. right) was obtained by adding (resp. subtracting) 0.2 to the center angle.
For each of the center, left and right images I obtained flipped images and used the negative of the steering angles.
The flipped image, for the sample center image above is:
![Flip Image][image2]
This step helped immensely to reduce the skew as shown in the histogram below.
![Aug Steering Hist][image6]


I converted the images returned by imread from BGR to RGB. This was an important point, that was
also pointed out by several people in the forums, mainly because while driving in the autonomous
mode, the model will be fed RGB images so in order to keep the input consistent this transformation
was necessary.

####3. Data Statistics and Model Tuning

The original data 7815 sample images from each of the cameras. 
Since I augmented the images by flippling the total number of input data images was 46890. This data was split in
the ratio 80%/20% into training set had 37512 images and the validation set which had 9378 images.
I trained the model for 5 epochs. I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.

I tried three different values of dropouts namely, 0.5, 0.6, and 0.7. The car was able
to successfully finish the loop with dropout of 0.6. However, visually, the driving was a little
smoother with dropout of 0.5.


####4. Conclusion

The project gave me a lot of experience in data gathering and preparation.
It was extremely critical to get the little details right.
One of the important ones was the conversion from BGR to RGB.
I trained the model for 5 epochs, however, there were instances when I was experimenting
that the model with smoothest driving did not have the best validation loss. 
Another interesting thing I observed was that using a smaller Dense layer after flattening
the input, I was able to get a much smoother drive through the first track.

The driving produced by the model tends zig-zag, especially around difficult curves.
I think a smoothing step where the steering angle is computed as an moving average of
certain frames would probably lead to better driving in the autonomous mode.

####5. Acknowledgement

I am thankful for Forum Mentors and participants for various helpful hints.
