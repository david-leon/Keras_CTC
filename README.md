# Keras MOD
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/daweileng/keras_MOD/blob/master/LICENSE)  
A modified fork of Keras, with base version 1.0.6  

------------------
## Purpose  
The modifications are mainly about transplanting a CTC (*Connectionist Temporal Classification*) implementation [see https://github.com/daweileng/Precise-CTC] into Keras. This feature has been missing by Keras for quite a long time.

------------------
## Functions
[1] Till now, the following train/test functions work well with CTC cost:
  * **train_on_batch()**
  * **test_on_batch()**
  * **predict_on_batch()**

The modification of '**fit()**' function is in progress, but no definte schedule.

To use CTC cost objective, set **loss = 'ctc_cost_for_train'** or **'ctc_cost_precise'** when compiling model.

[2] For now, only '**loss**' metric works with CTC cost. For accuracy evaluation, you need to do the decoding and calculate the CER outside Keras.

[3] '**Flatten**' class is modified to work with FCN (*Fully Convolutional Network*)

[4] '**Permute**' class is modified to work with FCRN (*Fully Convolutional Recurrent Network*)

[5] '**conv2d**'  class is modified to work with 'same' and 'full' mode

[6] '**Recurrent**' class is modified to work with bidirectional RNN

[7] Add support for multiple train/test/predict functions for 
  * **fit()** 
  * **evaluate()** 
  * **predict()** 
  * **train_on_batch()**
  * **test_on_batch()**
  * **predict_on_batch()**  
  
To use another train/test/predict function besides the 'default' one, just need to specify its 'name' parameter. Now we can build multi-task/multi-modal networks!

------------------
## Usage Guide (draft)
You can walk through all the added features of Keras-MOD with a toy demo, check '/ctc_demo/mnist_ctc_v9.py' for details.   
The toy demo uses mnist dataset to simulate digit sequence images, and use FCN + LSTM to recognize the digit texts.
```python
minNcharPerseq, maxNcharPerseq= 4, 6
print('Concatenating images')
X_train2, X_train2_mask, y_train2, y_train2_mask = mnist_concatenate_image(X_train, y_train, minNcharPerseq, maxNcharPerseq)
X_test2, X_test2_mask, y_test2, y_test2_mask     = mnist_concatenate_image(X_test, y_test, minNcharPerseq, maxNcharPerseq)
```
The code snipet above concatenates mnist digit images randomly, to simulate digit sequence images containing 4 ~ 6 number of digits.   
To recognize the digit texts from these images, we use FCN (Fully Convolutional Network) + LSTM (Long Short-Term Memory) RNN architecture. In the demo code there are two functions: **build_model** and **build_model_2**, the former one build a very simple model with only one CNN layer plus one LSTM layer:
```python
def build_model(feadim, Nclass, loss='ctc_cost_for_train', optimizer='Adadelta'):
    net_input = Input(shape=(1, feadim, None))
    net_cnn   = Convolution2D(1, 3, 3, activation='relu', border_mode='valid')(net_input)   # input shape = (samples, channels, rows, cols)
    net_reshape = Permute((3,2))(net_cnn)
    net_o1    = LSTM(100, return_sequences=True, activation='tanh')(net_reshape)            # input shape = (samples, timesteps, input_dim)
    net_o2    = TimeDistributed(Dense(Nclass + 1,  activation='softmax'))(net_o1)
    model     = Model(net_input, net_o2)
    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode='temporal')
    return model
```
The latter one build on the other hand a complicated model with upto 7 CNN layers, 4 max pooling layers, 3 batch normalizaiton layers plus 2 LSTM layers:
```python
def build_model_2(feadim, Nclass, loss='ctc_cost_for_train', optimizer='Adadelta', border_mode='same'):
    """
    Input shape: X.shape=(B, 1, rows, cols), GT.shape=(B, L)
    :param feadim: input feature dimension
    :param Nclass: class number
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
``` 
In the above code, two uni-directional LSTM layers with opposite directions are used instead of plain bi-directional LSTM. If you're interested in trying bi-directional RNN out, you can build a BLSTM layer as defined in **build_BLSTM_layer**
```python
def build_BLSTM_layer(inputdim, outputdim, return_sequences=True, activation='tanh'):
    net_input     = Input(shape=(None, inputdim))
    forward_lstm  = LSTM(output_dim=outputdim, return_sequences=return_sequences, activation=activation)(net_input)
    backward_lstm = LSTM(output_dim=outputdim, return_sequences=return_sequences, activation=activation, go_backwards=True, keep_time_order=False)(net_input)
    net_output = Merge(mode='concat')([forward_lstm, backward_lstm])
    BLSTM = Model(net_input, net_output)
    return BLSTM
``` 
and replace the LSTM layer in **build_model** and **build_model_2**.
        
From the above two models you now can see how 
  * 1) Fully convolutionaly network 
  * 2) End-to-end artecture of FCN + RNN
  * 3) bi-directional RNN    
    
can be built with Keras-MOD. 

 
------------------
### Why not pull request for Keras master branch?
Question may be asked why not make a pull request for fchollet's Keras repository. 

One reason is that CTC cost is rather different from the existing objective functions in Keras, it requires different masking mechanism and not directly reflects accuracy. To avoid mass modification of Keras' current masking mechanism, I override Keras' '**sample_weights**' variable for '**seq_mask**' of CTC and Keras' '**masks**' variable for '**sm_mask**' of CTC. This should not cause problem for other network architecture or objective functions, but I lack resources to give it a thorough test.

The other reason is that for now I only care about Theano implementations. So the modifications are targetted only on Theano backend.
