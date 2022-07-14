"""
Inception-v3 inflated 3D 

The code was adapted from https://github.com/keras-team/keras-applications/tree/master/keras_applications

# Reference
- [Rethinking the Inception Architecture for Computer Vision](
    http://arxiv.org/abs/1512.00567) (CVPR 2016)

"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.utils import get_source_inputs
from tensorflow.python.keras.utils import layer_utils
from tensorflow.keras import backend as K


def conv3d_bn(x,
              filters,
              num_frames,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1, 1),
              use_bias = False,
              use_activation_fn = True,
              use_bn = True,
              name=None):
    """Utility function to apply conv3d + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        num_frames: frames (time depth) of the convolution kernel.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        use_bias: use bias or not
        use_activation_fn: use an activation function or not.
        use_bn: use batch normalization or not.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv3D(
        filters, (num_frames, num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=conv_name)(x)

    if use_bn:
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 4
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if use_activation_fn:
        x = Activation('relu', name=name)(x)

    return x


def Inception_v3_Inflated3d(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                dropout_prob=0.0,
                endpoint_logit=True,
                classes=3):

    img_input = Input(shape=input_shape)
    channel_axis = 4
    
    x = conv3d_bn(img_input, 32, 3, 3, 3, strides=(2, 2, 2), padding='valid')
    x = conv3d_bn(x, 32, 3, 3, 3, padding='valid')
    x = conv3d_bn(x, 64, 3, 3, 3)
    x = layers.MaxPooling3D((1, 3, 3), strides=(1, 2, 2))(x)
    
    
    x = conv3d_bn(x, 80, 1, 1, 1, padding='valid')
    x = conv3d_bn(x, 192, 3, 3, 3, padding='valid')
    x = layers.MaxPooling3D((1, 3, 3), strides=(1, 2, 2))(x)
    
    # mixed 0: 35 x 35 x 256
    branch1x1 = conv3d_bn(x, 64, 1, 1, 1)

    branch5x5 = conv3d_bn(x, 48, 1, 1, 1)
    branch5x5 = conv3d_bn(branch5x5, 64, 5, 5, 5)

    branch3x3dbl = conv3d_bn(x, 64, 1, 1, 1)
    branch3x3dbl = conv3d_bn(branch3x3dbl, 96, 3, 3, 3)
    branch3x3dbl = conv3d_bn(branch3x3dbl, 96, 3, 3, 3)

    branch_pool = layers.AveragePooling3D((1, 3, 3),
                                          strides=(1, 1, 1),
                                          padding='same')(x)
    branch_pool = conv3d_bn(branch_pool, 32, 1, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv3d_bn(x, 64, 1, 1, 1)

    branch5x5 = conv3d_bn(x, 48, 1, 1, 1)
    branch5x5 = conv3d_bn(branch5x5, 64, 5, 5, 5)

    branch3x3dbl = conv3d_bn(x, 64, 1, 1, 1)
    branch3x3dbl = conv3d_bn(branch3x3dbl, 96, 3, 3, 3)
    branch3x3dbl = conv3d_bn(branch3x3dbl, 96, 3, 3, 3)

    branch_pool = layers.AveragePooling3D((1, 3, 3),
                                          strides=(1, 1, 1),
                                          padding='same')(x)
    branch_pool = conv3d_bn(branch_pool, 64, 1, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')
    
    
    # mixed 2: 35 x 35 x 288
    branch1x1 = conv3d_bn(x, 64, 1, 1, 1)

    branch5x5 = conv3d_bn(x, 48, 1, 1, 1)
    branch5x5 = conv3d_bn(branch5x5, 64, 5, 5, 5)

    branch3x3dbl = conv3d_bn(x, 64, 1, 1, 1)
    branch3x3dbl = conv3d_bn(branch3x3dbl, 96, 3, 3, 3)
    branch3x3dbl = conv3d_bn(branch3x3dbl, 96, 3, 3, 3)

    branch_pool = layers.AveragePooling3D((1, 3, 3),
                                          strides=(1, 1, 1),
                                          padding='same')(x)
    branch_pool = conv3d_bn(branch_pool, 64, 1, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')
    
    
    # mixed 3: 17 x 17 x 768
    branch3x3 = conv3d_bn(x, 384, 3, 3, 3, strides=(2, 2, 2), padding='valid')

    branch3x3dbl = conv3d_bn(x, 64, 1, 1, 1)
    branch3x3dbl = conv3d_bn(branch3x3dbl, 96, 3, 3, 3)
    branch3x3dbl = conv3d_bn(
        branch3x3dbl, 96, 3, 3, 3, strides=(2, 2, 2), padding='valid')

    branch_pool = layers.MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')
    
    
    
    # mixed 4: 17 x 17 x 768
    branch1x1 = conv3d_bn(x, 192, 1, 1, 1)

    branch7x7 = conv3d_bn(x, 128, 1, 1, 1)
    branch7x7 = conv3d_bn(branch7x7, 128, 1, 7, 7)
    branch7x7 = conv3d_bn(branch7x7, 192, 7, 1, 1)

    branch7x7dbl = conv3d_bn(x, 128, 1, 1, 1)
    branch7x7dbl = conv3d_bn(branch7x7dbl, 128, 7, 1, 1)
    branch7x7dbl = conv3d_bn(branch7x7dbl, 128, 1, 7, 7)
    branch7x7dbl = conv3d_bn(branch7x7dbl, 128, 7, 1, 1)
    branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 1, 7, 7)

    branch_pool = layers.AveragePooling3D((1, 3, 3),
                                          strides=(1, 1, 1),
                                          padding='same')(x)
    branch_pool = conv3d_bn(branch_pool, 192, 1, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')
    
    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv3d_bn(x, 192, 1, 1, 1)

        branch7x7 = conv3d_bn(x, 160, 1, 1, 1)
        branch7x7 = conv3d_bn(branch7x7, 160, 1, 7, 7)
        branch7x7 = conv3d_bn(branch7x7, 192, 7, 1, 1)

        branch7x7dbl = conv3d_bn(x, 160, 1, 1, 1)
        branch7x7dbl = conv3d_bn(branch7x7dbl, 160, 7, 1, 1)
        branch7x7dbl = conv3d_bn(branch7x7dbl, 160, 1, 7, 7)
        branch7x7dbl = conv3d_bn(branch7x7dbl, 160, 7, 1, 1)
        branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 1, 7, 7)

        branch_pool = layers.AveragePooling3D((1, 3, 3), 
                                              strides=(1, 1, 1), 
                                              padding='same')(x)
        branch_pool = conv3d_bn(branch_pool, 192, 1, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))
    
    # mixed 7: 17 x 17 x 768
    branch1x1 = conv3d_bn(x, 192, 1, 1, 1)

    branch7x7 = conv3d_bn(x, 192, 1, 1, 1)
    branch7x7 = conv3d_bn(branch7x7, 192, 1, 7, 7)
    branch7x7 = conv3d_bn(branch7x7, 192, 7, 1, 1)

    branch7x7dbl = conv3d_bn(x, 192, 1, 1, 1)
    branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 7, 1, 1)
    branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 1, 7, 7)
    branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 7, 1, 1)
    branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 1, 7, 7)

    branch_pool = layers.AveragePooling3D((1, 3, 3),
                                          strides=(1, 1, 1),
                                          padding='same')(x)
    branch_pool = conv3d_bn(branch_pool, 192, 1, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')
    
    
   # mixed 8: 8 x 8 x 1280
    branch3x3 = conv3d_bn(x, 192, 1, 1, 1)
    branch3x3 = conv3d_bn(branch3x3, 320, 3, 3, 3,
                          strides=(2, 2, 2), padding='valid')

    branch7x7x3 = conv3d_bn(x, 192, 1, 1, 1)
    branch7x7x3 = conv3d_bn(branch7x7x3, 192, 1, 7, 7)
    branch7x7x3 = conv3d_bn(branch7x7x3, 192, 7, 1, 1)
    branch7x7x3 = conv3d_bn(
        branch7x7x3, 192, 3, 3, 3, strides=(2, 2, 2), padding='valid')

    branch_pool = layers.MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8') 
    
    
    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv3d_bn(x, 320, 1, 1, 1)

        branch3x3 = conv3d_bn(x, 384, 1, 1, 1)
        branch3x3_1 = conv3d_bn(branch3x3, 384, 1, 3, 3)
        branch3x3_2 = conv3d_bn(branch3x3, 384, 3, 1, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv3d_bn(x, 448, 1, 1, 1)
        branch3x3dbl = conv3d_bn(branch3x3dbl, 384, 3, 3, 3)
        branch3x3dbl_1 = conv3d_bn(branch3x3dbl, 384, 1, 3, 3)
        branch3x3dbl_2 = conv3d_bn(branch3x3dbl, 384, 3, 1, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling3D(
            (1, 3, 3), strides=(1, 1, 1), padding='same')(x)
        branch_pool = conv3d_bn(branch_pool, 192, 1, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    '''
    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling3D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling3D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling3D()(x)
    '''
    
    h = int(x.shape[2])
    w = int(x.shape[3])
    #x = AveragePooling3D((2, h, w), strides=(1, 1, 1), padding='valid', name='global_avg_pool_ed')(x)
    #x = layers.GlobalAveragePooling3D(name='avg_pool')(x)
    # Create model.
    #model = Model(img_input, x, name='inception_v3')

    print("shape of global avg pool:",  x.shape)

    inputs = tf.keras.layers.Input(shape=input_shape)
    inputs_ = tf.keras.layers.Concatenate(axis=4)([inputs, inputs, inputs])
    x = tf.keras.layers.AveragePooling3D((1, 1, 1), 
                                              strides=(1, 1, 1),
                                              padding='valid')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(rate=dropout_prob)(x)
    x = tf.keras.layers.Dense(3)(x)
    out = tf.keras.layers.Activation('softmax')(x)

    #model  = tf.keras.Model(inputs = inputs, outputs = out)
    model = Model(img_input, out, name='i3d_inception')



    return model





