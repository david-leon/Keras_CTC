# coding:utf-8
# LSTM-CTC simulation experiment using augmented mnist data
# Created   :   1, 21, 2016
# Revised   :   1, 29, 2016
# Author    :  David Leon (Dawei Leng)
# All rights reserved
#------------------------------------------------------------------------------------------------
import os
if os.name =='nt':
    os.environ['THEANO_FLAGS'] ='floatX=float32,mode=FAST_RUN'
elif os.name == 'posix':
    os.environ['THEANO_FLAGS'] = "device=gpu0,floatX=float32,mode=FAST_RUN,gcc.cxxflags='-march=corei7',cuda.root= '/usr/local/cuda-7.0/'"

from keras.models import Model, Graph
from keras.layers import LSTM, TimeDistributed, Dense, Dropout, Input, Convolution2D, MaxPooling2D, Reshape, Permute, \
                         Merge, Activation, BatchNormalization

import numpy as np, time
import gzip, pickle, theano
from NN_auxiliary import pad_sequence_into_array
from CTC_utils import CTC
from theano import tensor

def build_BLSTM_layer(inputdim, outputdim, return_sequences=True, activation='tanh'):
    net_input     = Input(shape=(None, inputdim))
    forward_lstm  = LSTM(output_dim=outputdim, return_sequences=return_sequences, activation=activation)(net_input)
    backward_lstm = LSTM(output_dim=outputdim, return_sequences=return_sequences, activation=activation, go_backwards=True, keep_time_order=False)(net_input)
    net_output = Merge(mode='concat')([forward_lstm, backward_lstm])
    BLSTM = Model(net_input, net_output)
    return BLSTM

def build_model(feadim, Nclass, loss='ctc_cost_for_train', optimizer='Adadelta'):
    net_input = Input(shape=(1, feadim, None))
    net_cnn   = Convolution2D(1, 3, 3, activation='relu', border_mode='valid')(net_input)   # input shape = (samples, channels, rows, cols)
    net_reshape = Permute((3,2))(net_cnn)
    net_o1    = LSTM(100, return_sequences=True, activation='tanh')(net_reshape)            # input shape = (samples, timesteps, input_dim)
    # net_ad1   = LSTM(100, return_sequences=True, activation='tanh', go_backwards=True)(net_o1)
    # bilstm = build_BLSTM_layer(26, 100)
    # net_ad1 = bilstm(net_reshape)
    net_o2    = TimeDistributed(Dense(Nclass + 1,  activation='softmax'))(net_o1)
    model     = Model(net_input, net_o2)
    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode='temporal')
    return model

def build_model_2(feadim, Nclass, loss='ctc_cost_for_train', optimizer='Adadelta', border_mode='same'):
    """
    Input shape: X.shape=(B, 1, rows, cols), GT.shape=(B, L)
    :param feadim:
    :param Nclass:
    :param loss:
    :param optimizer:
    :return:
    """
    net_input = Input(shape=(1, feadim, None))
    cnn0   = Convolution2D( 64, 3, 3, border_mode=border_mode, activation='relu', name='cnn0')(net_input)
    pool0  = MaxPooling2D(pool_size=(2, 2), name='pool0')(cnn0)
    cnn1   = Convolution2D(128, 3, 3, border_mode=border_mode, activation='relu', name='cnn1')(pool0)
    pool1  = MaxPooling2D(pool_size=(2, 2), name='pool1')(cnn1)
    cnn2   = Convolution2D(256, 3, 3, border_mode=border_mode, activation='relu', name='cnn2')(pool1)
    BN0    = BatchNormalization(mode=0, axis=1, name='BN0')(cnn2)
    cnn3   = Convolution2D(256, 3, 3, border_mode=border_mode, activation='relu', name='cnn3')(BN0)
    pool2  = MaxPooling2D(pool_size=(2, 1), name='pool2')(cnn3)
    cnn4   = Convolution2D(512, 3, 3, border_mode=border_mode, activation='relu', name='cnn4')(pool2)
    BN1    = BatchNormalization(mode=0, axis=1, name='BN1')(cnn4)
    cnn5   = Convolution2D(512, 3, 3, border_mode=border_mode, activation='relu', name='cnn5')(BN1)
    pool3  = MaxPooling2D(pool_size=(2, 1), name='pool3')(cnn5)
    cnn6   = Convolution2D(1,   3, 3, border_mode=border_mode, activation='relu', name='cnn6')(pool3)
    BN2    = BatchNormalization(mode=0, axis=1, name='BN2')(cnn6)
    net_reshape = Permute((3, 2), name='net_reshape')(BN2)
    lstm0  = LSTM(100, return_sequences=True, activation='tanh', name='lstm0')(net_reshape)
    lstm1  = LSTM(100, return_sequences=True, activation='tanh', go_backwards=True, keep_time_order=True, name='lstm1')(lstm0)
    dense0 = TimeDistributed(Dense(Nclass + 1, activation='softmax', name='dense0'))(lstm1)
    model  = Model(net_input, dense0)
    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode='temporal')
    return model


def mnist_concatenate_image(Xin, yin, minNcharPerseq=3, maxNcharPerseq=6):
    """
    Concatenate single char images to simulate character sequence image
    :param Xin: X_train or X_test, with shape (Nsample, Ncolumn, Nrow)
    :param yin: y_train or y_test, with shape (Nsample, )
    :param minNcharPerseq: minimum char number per sequence
    :param maxNcharPerseq: maximum char number per sequence
    :return: X_aug (B, T, D), X_aug_mask (B, T), y_aug (B, L), y_aug_mask (B, L)
    """
    Nsample = Xin.shape[0]
    RandomWalk = np.random.choice(Nsample, Nsample * maxNcharPerseq, replace = True)
    X_aug_list = []
    y_aug_list = []
    idx = 0
    for i in range(Nsample):
        seqlen = np.random.randint(minNcharPerseq, maxNcharPerseq+1)
        x = Xin[RandomWalk[idx], :, :].T
        y = yin[RandomWalk[idx]] * np.ones([seqlen], dtype=yin.dtype)
        idx += 1
        for j in range(seqlen-1):
            x = np.concatenate([x, Xin[RandomWalk[idx], :, :].T], axis=0)
            y[j+1] = yin[RandomWalk[idx]]
            idx += 1

        X_aug_list.append(x)
        y_aug_list.append(y)

    X_aug, X_aug_mask = pad_sequence_into_array(X_aug_list)
    y_aug, y_aug_mask = pad_sequence_into_array(y_aug_list)
    return X_aug, X_aug_mask, y_aug, y_aug_mask

def convert_gt_from_array_to_list(gt_batch, gt_batch_mask=None):
    """
    Convert groundtruth from ndarray to list
    :param gt_batch: ndarray (B, L)
    :param gt_batch_mask: ndarray (B, L)
    :return: gts <list of size = B>
    """
    B, L = gt_batch.shape
    gt_batch = gt_batch.astype('int')
    gts = []
    for i in range(B):
        if gt_batch_mask is None:
            l = L
        else:
            l = int(gt_batch_mask[i,:].sum())
        gts.append(gt_batch[i,:l].tolist())
    return gts

def _change_input_shape(floatx='float32'):
    x = tensor.tensor3('input', dtype=floatx)
    y = x.dimshuffle((0, 'x', 2, 1))
    f = theano.function([x], y, allow_input_downcast=True)
    return f


if __name__ == '__main__':
    np.random.seed(1337) # for reproducibility
    model_ver = '2'

    print("model compiling...")
    loss = 'ctc_cost_for_train'
    optimizer = 'Adadelta'
    if model_ver == '1':
        model = build_model(feadim=28, Nclass=10, loss=loss, optimizer=optimizer)
    elif model_ver == '2':
        model = build_model_2(feadim=28, Nclass=10, loss=loss, optimizer=optimizer)
    else:
        raise ValueError('model_ver should be 1 or 2')

    reshape_func = _change_input_shape()
    print('reshape_func compiled')

    with gzip.open('mnist_float64.gpkl', 'rb') as f:
        X_train, y_train, X_test, y_test = pickle.load(f)
        f.close()

    minNcharPerseq, maxNcharPerseq= 4, 6
    print('Concatenating images')
    X_train2, X_train2_mask, y_train2, y_train2_mask = mnist_concatenate_image(X_train, y_train, minNcharPerseq, maxNcharPerseq)
    X_test2, X_test2_mask, y_test2, y_test2_mask = mnist_concatenate_image(X_test, y_test, minNcharPerseq, maxNcharPerseq)

    print('X_train2 shape:', X_train2.shape)                    # (B, T, D)
    print('y_train2 shape:', y_train2.shape)                    # (B, L)


    Nclass = 10
    B, T, D = X_train2.shape   # D = 28
    L = y_train2.shape[1]

    print("model training")
    max_epoch_num = 100
    batch=32*2
    for epoch in range(max_epoch_num):
        total_seqlen, total_ed = 0.0, 0.0
        batches = range(0, B, batch)
        shuffle = np.random.choice(batches, size=len(batches), replace=False)
        n = 0
        epoch_time0 = time.time()
        for i in shuffle:
            time0 = time.time()
            traindata0 = X_train2[i:i+batch, :, :]
            traindata = reshape_func(traindata0)
            if model_ver == '1':
                traindata_mask = X_train2_mask[i:i+batch, :-2]
            else:
                traindata_mask = X_train2_mask[i:i+batch, 0::16][:,:-1]
            gt = y_train2[i:i+batch, :]
            gt_mask = y_train2_mask[i:i+batch, :]
            ctcloss, score_matrix = model.train_on_batch(x=traindata, y=gt, sample_weight=gt_mask,
                                                         sm_mask=traindata_mask, return_sm=True)
            print('ctcloss = ', ctcloss)
            resultseqs = CTC.best_path_decode_batch(score_matrix, traindata_mask)
            targetseqs = convert_gt_from_array_to_list(gt, gt_mask)
            CER_batch, ed_batch, seqlen_batch = CTC.calc_CER(resultseqs, targetseqs)
            total_seqlen += seqlen_batch
            total_ed += ed_batch
            CER = total_ed / total_seqlen * 100.0
            n += 1
            time1 = time.time()
            print('epoch = %d, CER = %0.2f, CER_batch = %0.2f, time = %0.2fs, progress = %0.2f%%' % (epoch, CER, CER_batch, (time1-time0), n / len(batches) * 100.0))
        epoch_time1 = time.time()
        print('epoch = %d, CER = %0.2f, time = %0.2fmins' % (epoch, CER, (epoch_time1-epoch_time0)/60))
