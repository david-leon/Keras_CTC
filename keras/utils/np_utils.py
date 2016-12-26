from __future__ import absolute_import
import numpy as np
import scipy as sp
from six.moves import range
from six.moves import zip
from .. import backend as K


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes

    # Returns
        A binary matrix representation of the input.
    '''
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def binary_logloss(p, y):
    epsilon = 1e-15
    p = sp.maximum(epsilon, p)
    p = sp.minimum(1-epsilon, p)
    res = sum(y * sp.log(p) + sp.subtract(1, y) * sp.log(sp.subtract(1, p)))
    res *= -1.0/len(y)
    return res


def multiclass_logloss(P, Y):
    npreds = [P[i][Y[i]-1] for i in range(len(Y))]
    score = -(1. / len(Y)) * np.sum(np.log(npreds))
    return score


def accuracy(p, y):
    return np.mean([a == b for a, b in zip(p, y)])


def probas_to_classes(y_pred):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        return categorical_probas_to_classes(y_pred)
    return np.array([1 if p > 0.5 else 0 for p in y_pred])


def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)


def convert_kernel(kernel, dim_ordering='default'):
    '''Converts a kernel matrix (Numpy array)
    from Theano format to TensorFlow format
    (or reciprocally, since the transformation
    is its own inverse).
    '''
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if not 4 <= kernel.ndim <= 5:
        raise ValueError('Invalid kernel shape:', kernel.shape)

    slices = [slice(None, None, -1) for i in range(kernel.ndim)]
    no_flip = (slice(None, None), slice(None, None))
    if dim_ordering == 'th':  # (out_depth, input_depth, ...)
        slices[:2] = no_flip
    elif dim_ordering == 'tf':  # (..., input_depth, out_depth)
        slices[-2:] = no_flip
    else:
        raise ValueError('Invalid dim_ordering:', dim_ordering)

    return np.copy(kernel[slices])


def conv_output_length(input_length, filter_size, border_mode, stride, dilation=1):
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid', 'full'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif border_mode == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


def conv_input_length(output_length, filter_size, border_mode, stride):
    if output_length is None:
        return None
    assert border_mode in {'same', 'valid', 'full'}
    if border_mode == 'same':
        pad = filter_size // 2
    elif border_mode == 'valid':
        pad = 0
    elif border_mode == 'full':
        pad = filter_size - 1
    return (output_length - 1) * stride - 2 * pad + filter_size
