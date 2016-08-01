from __future__ import absolute_import
import numpy as np
from . import backend as K


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return K.categorical_crossentropy(y_pred, y_true)


def sparse_categorical_crossentropy(y_true, y_pred):
    '''expects an array of integer classes.
    Note: labels shape must have the same number of dimensions as output shape.
    If you get a shape error, add a length-1 dimension to labels.
    '''
    return K.sparse_categorical_crossentropy(y_pred, y_true)


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)


# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')

#------------------------------     for CTC cost      -------------------------------------------------#
from .ctc_theano import CTC_precise, CTC_for_train
import theano.tensor as tensor

def ctc_cost_precise(seq, sm, seq_mask=None, sm_mask=None):
    """
    seq (B, L), sm (B, T, C+1), seq_mask (B, L), sm_mask (B, T)
    Compute CTC cost, using only the forward pass
    :param queryseq: (L, B)
    :param scorematrix: (T, C+1, B)
    :param queryseq_mask: (L, B)
    :param scorematrix_mask: (T, B)
    :param blank_symbol: scalar
    :return: negative log likelihood averaged over a batch
    """
    queryseq = seq.T
    scorematrix = sm.dimshuffle(1, 2, 0)
    if seq_mask is None:
        queryseq_mask = None
    else:
        queryseq_mask = seq_mask.T
    if sm_mask is None:
        scorematrix_mask = None
    else:
        scorematrix_mask = sm_mask.T

    return CTC_precise.cost(queryseq, scorematrix, queryseq_mask, scorematrix_mask)

def ctc_cost_for_train(seq, sm, seq_mask=None, sm_mask=None):
    """
    seq (B, L), sm (B, T, C+1), seq_mask (B, L), sm_mask (B, T)
    Compute CTC cost, using only the forward pass
    :param queryseq: (L, B)
    :param scorematrix: (T, C+1, B)
    :param queryseq_mask: (L, B)
    :param scorematrix_mask: (T, B)
    :param blank_symbol: scalar
    :return: negative log likelihood averaged over a batch
    """
    queryseq = tensor.addbroadcast(seq.T)
    scorematrix = sm.dimshuffle(1, 2, 0)
    if seq_mask is None:
        queryseq_mask = None
    else:
        queryseq_mask = seq_mask.T
    if sm_mask is None:
        scorematrix_mask = None
    else:
        scorematrix_mask = sm_mask.T

    return CTC_for_train.cost(queryseq, scorematrix, queryseq_mask, scorematrix_mask)

def ctc_decode(y_hat, y_hat_mask=None):
    scorematrix = y_hat.dimshuffle(1, 2, 0)
    if y_hat_mask is None:
        scorematrix_mask = None
    else:
        scorematrix_mask = y_hat_mask.dimshuffle(1, 0)
    blank_symbol = y_hat.shape[2] - 1
    resultseq, resultseq_mask = CTC_precise.best_path_decode(scorematrix, scorematrix_mask, blank_symbol)
    return resultseq, resultseq_mask

def ctc_CER(resultseq, targetseq, resultseq_mask=None, targetseq_mask=None):
    return CTC_precise.calc_CER(resultseq, targetseq.T, resultseq_mask, targetseq_mask)

if __name__ == '__main__':
    fn = get('ctc_cost_for_train')
    print(fn)