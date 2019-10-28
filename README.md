# TrainingDataSynthesizer

The Synthesizer.py file contains the code implementation of the algorithm in the paper.

The main.py imports the Classifier and Synthesizer and carries out the necessary computations. Run this to get synthesized training data. The main.py contains code for two datasets. Comment out the sections as required.

The dataset and code for the Neural Network for MNIST can be found in the following link. Steps to generate the pkl file for faster reload are also given here.
 https://www.python-course.eu/neural_network_mnist.php


 # MNIST Synthesizer

 As mentioned in the paper, one way to generate data is to start with a base dataset (Page 5 in the paper under heading 'Noisy real data') and then manipulate attributes using this dataset. Since this dataset is a collection of numbers from 0 to 9, we can assume that the attacker can get enough data records (by manually acquiring 28*28 pixel images of the digits). We can create this base dataset by using the keras library as mentioned here (https://medium.com/@ashok.tankala/build-the-mnist-model-with-your-own-handwritten-digits-using-tensorflow-keras-and-python-f8ec9f871fd3). 
 
 For simplicity, the base dataset used in this example is from the test data from MNIST dataset itself. 

 In the randomisation function here, we randomly select gray pixels from the image. For each randomly selected gray pixel, we change one of it's neighbouring pixels. The distance from the pixel is defined by the stride. The larger the stride, the more the variance from the original image. This process ensures that the randomly selected pixel is more localised, failing which it is near impossible to generate records with high confidence values for one of the classes (in this case we would usually get noisy unintelligible images in each iteration).

# Link to the paper:

https://arxiv.org/pdf/1610.05820.pdf
