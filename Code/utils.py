import scipy.io
import imageio
import numpy as np
import tensorflow as tf

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 300
COLOR_CHANNELS = 3
NOISE_RATIO = 0.6
MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
    
def load_vgg_model(path):
    """
    Loads the model from the path and
    creats a dictionary to hold the layers and their weights
    """
    model = scipy.io.loadmat(path)
    layers = model['layers']
    
    def _weights(layer):
        """
        extracts and returns the weights and bias parameters 
        of the passed layer
        """
        wb = layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        return W, b

    def _conv2d(prev_layer, layer):
        """
        Returns the result of application of a 2D Convolution on the 'prev_layer' with the filters of 'layer'
        """
        W, b = _weights(layer)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filters=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer):
        """
        Returns the combination of ReLU layer and a 2D Convolutional layer, applied on 'prev_layer'
        """
        return tf.nn.relu(_conv2d(prev_layer, layer))

    def _avgpool(prev_layer):
        """
        Returns result after applying Average Pooling on 'prev_layer'
        """
        return tf.nn.avg_pool2d(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0)
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2)
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5)
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7)
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10)
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12)
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14)
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16)
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19)
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21)
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23)
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25)
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28)
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30)
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32)
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34)
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    
    return graph

def generate_noise_image(content_image, noise_ratio = NOISE_RATIO):
    
    noise_image = np.random.uniform(-20, 20, (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS))
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image


def reshape_and_normalize_image(image):
    image = np.reshape(image, ((1,) + image.shape))
    image = image - MEANS
    return image

def loadimg(path):
    image = imageio.imread(path)
    assert image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)
    return image