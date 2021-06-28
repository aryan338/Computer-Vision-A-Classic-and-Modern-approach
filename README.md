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

## Model
Following are the steps followed:
1)  Load the images
2)  Randomly initialize the image to be generated
3)  Load the VGG-19 model
4)  Calculate the required activations for the content image C, the style image S, and the generated image G, for the chosen layer(s).
5)  Iteratively, calculate the content loss and the style loss by following the above equations. Use them to compute the total loss and update the generated image so as to minimize the loss.

## Costs
We will define two different cost functions, one each corresponding to the ‘Content Image’ and the ‘Style Image’ respectively

### Content Cost
Here, we will calculate the closeness of the content in the generated image with the ‘Content Image’. We will accomplish that by simply calculating the Euclidean distances between their matrix representations, i.e., by calculating the Frobenius Norm of their element-wise difference matrix. So, mathematically, our cost function will look like the following - 

Here, *n<sub>H</sub>, n<sub>W</sub>, n<sub>C</sub>* are the height, width, and number of channels in the chosen layer. *a<sup>(G)</sup>* and *a<sup>(C)</sup>* are the activations at the chosen layer with inputs as the generated image G and the content image C.

### Style Cost
Here, the cost computation is slightly more engaging. We take the help of gram matrices for the cost calculation. In linear algebra, the gram matrix *G<sub>gram</sub>* for a set of vectors *(v<sub>1</sub>,v<sub>2</sub>,...,v<sub>n</sub>)* is the matrix of dot products, where *G<sub>gram</sub>(i,j)= v<sub>i</sub><sup>T</sup>v<sub>j</sub>*.
For calculating the style cost, the following steps are taken -
1)  The activations of the chosen layer (each with the generated image G and the style image S as inputs) are reshaped into *n<sub>C</sub> x (n<sub>H</sub> x n<sub>W</sub>)* matrices.
2)  Then, the Gram matrices of the above-unrolled matrices are found out.
3)  The style cost is calculated using the Frobenius Norm of their element-wise difference matrix.

Mathematically, the equation for the style cost looks like the following - 

Here, *n<sub>H</sub>, n<sub>W</sub>, n<sub>C</sub>* are the height, width, and number of channels in the chosen layer. *G<sup>(G)</sup><sub>gram</sub>* and *G<sup>(S)</sup><sub>gram</sub>* are the gram matrices of the activation at the chosen layer with inputs as the generated image G and the style image S.

Note - For calculating the content cost, the activations from one layer are deemed enough. But for computing the style cost, taking the total cost from different layers will be better for the model’s performance.

### Total Cost
The total cost can be simply calculated by a weighted sum of the content cost and the style cost - 

where *&alpha;* and *&beta;* can be set manually and treated as hyper-parameters that control the relative weighting between the content and the style.


Team: Aditya Prakash, Aryan Mundada, Harsh Kumar and Imad Khan
Offered by : Shivanshu Tyagi and Akshay Gupta