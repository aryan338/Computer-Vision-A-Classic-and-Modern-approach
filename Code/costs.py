import tensorflow as tf

def compute_content_cost(C, G):
    """
    Takes two arguments C, G which represent the hidden layer activations of the content image and the generated image respectively.
    C and G are tensors of dimension (1, h, w, c)
    Returns the content cost.
    """
    m, h, w, c = G.shape
    return tf.reduce_sum(tf.square(G - C))/(4.0 * int(h) * int(w) * int(c))


def gram_matrix(A):
    """
    Takes a matrix A of shape (c, h * w)
    Returns the gram matrix of A, of shape (c, c)
    """
    return tf.matmul(A, tf.transpose(A))


def compute_style_cost(S, G):
    """
    Takes two arguments S, G which represent the hidden layer activations of the style image and the generated image respectively.
    S and G are tensors of dimension (1, h, w, c)
    Returns the style cost.
    """
    m, h, w, c = G.shape
    S = tf.reshape(S, [h * w, c])
    S = tf.transpose(S)
    G = tf.reshape(G, [h * w, c])
    G = tf.transpose(G)

    GS = gram_matrix(S)
    GG = gram_matrix(G)

    return tf.reduce_sum(tf.square(GS - GG))/(4.0*(int(h)**2)*(int(w)**2)*(int(c)**2))


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    To compute the total cost 
    
    Takes 4 arguments, the content cost, the style cost,
    and the hyperparameters, alpha and beta, weighting the importance of the content cost and style cost respectively
    Returns the total cost 
    """
    return alpha * J_content + beta * J_style