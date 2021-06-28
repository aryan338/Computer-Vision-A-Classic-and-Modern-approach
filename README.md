# Neural Style Transfer
This repo contains files and codes related to the CV project offered by Stamatics, IITK in summers of 2021.

## About our Project
Neural Style Transfer is a simple application of Convolutional Neural Networks and Transfer Learning. In this application, we have two images, one labeled as a ‘Content Image’ and the other labeled as a ‘Style Image’ and our aim is to generate an image that consists of all the content in the ‘Content Image’, styled according to the ‘Style Image’. The power of Convolutional Neural Networks can be taken advantage of for this task. 

## Pre-trained Data
For the model for generating the image, we shall take help from some Transfer Learning basics, by using a pre-trained ‘VGG-19 ImageNet’ network. We shall use this pre-trained model to update a randomly initialised image according to the constraints that we define and implement some custom loss functions to update the model parameters accordingly.

## Data for Execution
This task requires only 2 images -
1)  Content Image (C) - This image contains the primary content to which some particular style needs to be applied. It is usually the semantically more informative of the two images.
2)  Style Image (S) - This image is expected to be less informative and more abstract as it will dictate the style of our generated image.

Apart from the images for the training and updating of our model parameters, we also need the pre-trained VGG-19 model itself to train. This is can be easily obtained via the tensorflow library.

## Output
Generated Image (G) - This is the image that we aim to produce. This image should incorporate maximum content from content image C and at the same time, have the style as similar as possible to style image S.

Team: Aditya Prakash, Aryan Mundada, Harsh Kumar and Imad Khan
Offered by : Shivanshu Tyagi and Akshay Gupta