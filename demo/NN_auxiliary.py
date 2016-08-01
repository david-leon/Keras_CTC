# coding:utf-8
# Collection of auxiliary functions for DL
# Created   :   1, 21, 2016
# Revised   :   1, 22, 2016
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import numpy as np

def pad_sequence_into_array(Xs, maxlen=None, truncating='post', padding='post', value=0.):
    """
    Padding sequence (list of numpy arrays) into an numpy array
    :param Xs: list of numpy arrays. The arrays must have the same shape except the first dimension.
    :param maxlen: the allowed maximum of the first dimension of Xs's arrays. Any array longer than maxlen is truncated to maxlen
    :param truncating: = 'pre'/'post', indicating whether the truncation happens at either the beginning or the end of the array (default)
    :param padding: = 'pre'/'post',indicating whether the padding happens at either the beginning or the end of the array (default)
    :param value: scalar, the padding value, default = 0.0
    :return: Xout, the padded sequence (now an augmented array with shape (Narrays, N1stdim, N2nddim, ...)
    :return: mask, the corresponding mask, binary array, with shape (Narray, N1stdim)
    """
    Nsamples = len(Xs)
    if maxlen is None:
        lengths = [s.shape[0] for s in Xs]    # 'sequences' must be list, 's' must be numpy array, len(s) return the first dimension of s
        maxlen = np.max(lengths)

    Xout = np.ones(shape = [Nsamples, maxlen] + list(Xs[0].shape[1:]), dtype=Xs[0].dtype) * np.asarray(value, dtype=Xs[0].dtype)
    Mask = np.zeros(shape = [Nsamples, maxlen], dtype=Xout.dtype)
    for i in range(Nsamples):
        x = Xs[i]
        if truncating == 'pre':
            trunc = x[-maxlen:]
        elif truncating == 'post':
            trunc = x[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)
        if padding == 'post':
            Xout[i, :len(trunc)] = trunc
            Mask[i, :len(trunc)] = 1
        elif padding == 'pre':
            Xout[i, -len(trunc):] = trunc
            Mask[i, -len(trunc):] = 1
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return Xout, Mask



if __name__ == '__main__':
    INFO = ['This is a collection of auxiliary functions for DL.\n',
            'Author: David Leon\n',
            'All rights reserved\n']
    print(*INFO)

    # x1 = np.array([1,2,3])
    # x2 = np.array([1,2,3,4])
    # x3 = np.array([5,4,3,2,1])
    x1, x2, x3 = np.random.rand(2, 4), np.random.rand(3, 4), np.random.rand(5,4)

    Xout, mask = pad_sequence_into_array([x1,x2,x3])
    print(Xout)
    print(mask)
    print(Xout.shape)
    print(mask.shape)