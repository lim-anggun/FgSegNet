#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:20:23 2018

@author: longang
"""

from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import conv_utils
from keras.legacy import interfaces

# version 2.1.x has now base_layer class, so we need to import
if keras.__version__<'2.2':
    from keras.engine.topology import InputSpec
else:
    from keras.engine.base_layer import InputSpec

import tensorflow as tf
import numpy as np

class MyUpSampling2D(Layer):
    
    @interfaces.legacy_upsampling2d_support
    def __init__(self, size=(2, 2), num_pixels = (0, 0), data_format='channels_last', method_name='FgSegNet_M', **kwargs):
        super(MyUpSampling2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)
        self.num_pixels = num_pixels
        self.method_name = method_name
        assert method_name in ['FgSegNet_M', 'FgSegNet_S', 'FgSegNet_v2'], 'Provided method_name is incorrect.'

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] + self.num_pixels[0] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] + self.num_pixels[1] if input_shape[2] is not None else None
            return (input_shape[0], height, width, input_shape[3])
        
        else:
            raise ValueError('Invalid data_format:', self.data_format)
        
    def call(self, inputs):
        return resize_images(inputs, self.size[0], self.size[1], self.data_format, self.num_pixels, self.method_name)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format,
                  'num_pixels': self.num_pixels}
        base_config = super(MyUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# keras func: tensorflow_backend
def resize_images(x, height_factor, width_factor, data_format, num_pixels=None, method_name=None):
    """Resizes the images contained in a 4D tensor.
    # Arguments
        x: Tensor or variable to resize.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: string, `"channels_last"`
    # Returns
        A tensor.
    # Raises
        ValueError: if `data_format` is neither `"channels_last"`.
    """
    
    if data_format == 'channels_last':
        original_shape = K.int_shape(x) # (None, 67, 90, 512)
        new_shape = tf.shape(x)[1:3] # (67, 90, 512)
        
        #print(new_shape.get_shape().as_list())
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        if(num_pixels is not None):
           new_shape += tf.constant(np.array([num_pixels[0], num_pixels[1]]).astype('int32'))
        
        if method_name in ['FgSegNet_M', 'FgSegNet_S']:
            x = tf.image.resize_nearest_neighbor(x, new_shape)
        else: # FgSegNet_v2
            x = tf.image.resize_bilinear(x, new_shape)
        
        if(num_pixels is not None):
            x.set_shape((None, original_shape[1] * height_factor + num_pixels[0] if original_shape[1] is not None else None,
                     original_shape[2] * width_factor + num_pixels[1] if original_shape[2] is not None else None, None))
        else:
            x.set_shape((None, original_shape[1] * height_factor if original_shape[1] is not None else None,
                     original_shape[2] * width_factor if original_shape[2] is not None else None, None))
        return x
    else:
        raise ValueError('Invalid data_format:', data_format)
