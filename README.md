## CS4100 project

### Tech
We made a classifier for healthy and unhealthy apple tree leaves. 
Idea came from a kaggle competition, data from there as well. 
Model is based on [this keras tutorial](https://keras.io/examples/vision/image_classification_from_scratch/).
It is a deep convolutional neural network. All images are included in this repo.
One change that was made, is reduction of number of classes from 4 to 2 
(healthy and unhelthy, instead of multiple ones present in the competition).
Before run through the model, images are downscaled to 190x135 (preserving the 
original image ratio). Color is an important variable here, since removing 
it reduces accuracy on test/validation set to ~80% from 95%+. And on the setup this
was tested on, it decreased one epoch time by only 1s (from 9-10s avage).

### How to run it
Clone it and run the plant_nn.py. Tested to work on python 3.9, and tensorflow 2.5. 

### References
Data:
Thapa, R., Zhang, K., Snavely, N., Belongie, S., and Khan, A.. 2020. 
The Plant Pathology Challenge 2020 data set to classify foliar 
disease of apples. Applications in Plant Sciences 8( 9): e11390. 

[link](https://bsapubs.onlinelibrary.wiley.com/doi/10.1002/aps3.11390)

Kaggle competition:

[link](https://www.kaggle.com/c/plant-pathology-2020-fgvc7)

