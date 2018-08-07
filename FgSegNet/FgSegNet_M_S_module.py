#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 22:57:12 2017

@author: longang
"""

import keras
from keras.models import Model
from keras.layers import Activation, Input, Dropout, BatchNormalization, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers

from my_upsampling_2d import MyUpSampling2D
import keras.backend as K
import tensorflow as tf

def loss(y_true, y_pred):
    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc(y_true, y_pred):
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def loss2(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc2(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


class FgSegNet_M_S_module(object):
    
    def __init__(self, lr, reg, img_shape, scene, vgg_weights_path):
        self.lr = lr
        self.reg = reg
        self.img_shape = img_shape
        self.scene = scene
        self.vgg_weights_path = vgg_weights_path

    def VGG16(self, x): 
        
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Dropout(0.5, name='dr1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Dropout(0.5, name='dr2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Dropout(0.5, name='dr3')(x)
        
        return x
        
    def transposedConv(self, x):
        
        # block 5
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block5_tconv1', 
                                                kernel_regularizer=regularizers.l2(self.reg))(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='block5_tconv2')(x)
        x = Conv2DTranspose(512, (1, 1), activation='relu', padding='same', name='block5_tconv3')(x)
        
        # block 6
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block6_tconv1', 
                                                kernel_regularizer=regularizers.l2(self.reg))(x)
        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), activation='relu', padding='same', name='block6_tconv2')(x)
        x = Conv2DTranspose(256, (1, 1), activation='relu', padding='same', name='block6_tconv3')(x)
        
        # block 7
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block7_tconv1', 
                                                kernel_regularizer=regularizers.l2(self.reg))(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='block7_tconv2')(x)
        x = Conv2DTranspose(128, (1, 1), activation='relu', padding='same', name='block7_tconv3')(x)
        
        # block 8
        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), activation='relu', padding='same', name='block8_conv1', 
                                                kernel_regularizer=regularizers.l2(self.reg))(x)
        
        # block 9
        x = Conv2DTranspose(1, (1, 1), padding='same', name='block9_conv1')(x)
        x = Activation('sigmoid')(x)
        
        return x

    def initModel_M(self, dataset_name):
        assert dataset_name in ['CDnet', 'SBI', 'UCSD'], 'dataset_name must be either one in ["CDnet", "SBI", "UCSD"]]'
        assert len(self.img_shape)==3
        h, w, d = self.img_shape
        
        input_1 = Input(shape=(h, w, d), name='ip_scale1')
        vgg_layer_output = self.VGG16(input_1)
        shared_model = Model(inputs=input_1, outputs=vgg_layer_output, name='shared_model')
        shared_model.load_weights(self.vgg_weights_path, by_name=True)
        
        unfreeze_layers = ['block4_conv1','block4_conv2', 'block4_conv3']
        for layer in shared_model.layers:
            if(layer.name not in unfreeze_layers):
                layer.trainable = False
                
                
        # Scale 1
        x1 = shared_model.output
        # Scale 2
        input_2 = Input(shape=(int(h/2), int(w/2), d), name='ip_scale2')
        x2 = shared_model(input_2)
        x2 = UpSampling2D((2,2))(x2)
        # Scale 3
        input_3 = Input(shape=(int(h/4), int(w/4), d), name='ip_scale3')
        x3 = shared_model(input_3)
        x3 = UpSampling2D((4,4))(x3)
        
        
        if dataset_name=='CDnet':
            # Scale 1
            x1_ups = {'streetCornerAtNight':(0,1), 'tramStation':(1,0), 'turbulence2':(1,0)}
            if(self.scene=='wetSnow'):
                x1 = Cropping2D(cropping=((1, 2),(0, 0)))(x1)
            elif(self.scene=='skating'):
                x1 = Cropping2D(cropping=((1, 1),(1, 2)))(x1)
            else:
                for key, val in x1_ups.items():
                    if self.scene==key:
                        # upscale by adding number of pixels to each dim.
                        x1 = MyUpSampling2D(size=(1,1), num_pixels=val)(x1)
                        break
            
            # Scale 2
            x2_ups = {'tunnelExit_0_35fps':(0,1),'tramCrossroad_1fps':(1,0),'bridgeEntry':(1,1),
                      'busyBoulvard':(1,0),'fluidHighway':(0,1),'streetCornerAtNight':(1,1), 
                      'tramStation':(2,0),'winterStreet':(1,0),'twoPositionPTZCam':(1,0),
                      'peopleInShade':(1,1),'turbulence2':(1,1),'turbulence3':(1,0),
                      'skating':(1,1), 'wetSnow':(0,0)}
            for key, val in x2_ups.items():
                if self.scene == key and self.scene in ['skating', 'wetSnow']:
                    x2 = Cropping2D(cropping=((1, 1), val))(x2)
                    break
                elif self.scene==key:
                    x2 = MyUpSampling2D(size=(1, 1), num_pixels=val)(x2)
                    break
            
            # Scale 3
            x3_ups = {'tunnelExit_0_35fps':(2,3),'tramCrossroad_1fps':(3,0),'bridgeEntry':(3,1,),
                      'busyBoulvard':(3,0),'fluidHighway':(0,3),'streetCornerAtNight':(1,1),
                      'tramStation':(2,0),'winterStreet':(1,0),'twoPositionPTZCam':(1,2),
                      'peopleInShade':(1,3),'turbulence2':(3,1),'turbulence3':(1,0),
                      'office':(0,2), 'pedestrians':(0,2), 'bungalows':(0,2), 'busStation':(0,2)}
            
            for key, val in x3_ups.items():
                if self.scene==key:
                    x3 = MyUpSampling2D(size=(1,1), num_pixels=val)(x3)
                    break
                
        elif dataset_name=='SBI':
            if(self.scene=='Board'):
                x2 = MyUpSampling2D(size=(1,1), num_pixels=(1,0))(x2)
                x3 = MyUpSampling2D(size=(1,1), num_pixels=(1,2))(x3)
            elif(self.scene=='CaVignal'):
                x3 = MyUpSampling2D(size=(1,1), num_pixels=(2,2))(x3)
            elif(self.scene=='Foliage'):
                x3 = MyUpSampling2D(size=(1,1), num_pixels=(0,2))(x3)
            elif(self.scene=='Toscana'):
                x3 = MyUpSampling2D(size=(1,1), num_pixels=(2,0))(x3)
                
        elif dataset_name=='UCSD':
            x2_ups = {'birds':(1,0),'chopper':(1,0),'flock':(1,0),'freeway':(1,1),
                      'hockey':(1,1),'jump':(1,0),'landing':(1,1),'ocean':(1,1),
                      'rain':(1,1),'skiing':(1,0),'surf':(1,0),'traffic':(1,1),'zodiac':(1,1)}
            x3_ups = {'birds':(3,0),'boats':(0,2),'chopper':(3,0),'cyclists':(2,0),
                      'flock':(3,0),'freeway':(3,3),'hockey':(3,1),'jump':(3,0),
                      'landing':(3,1),'ocean':(1,3),'peds':(2,2),'rain':(1,1),
                      'skiing':(3,0),'surf':(3,0),'surfers':(0,2),'traffic':(1,1),'zodiac':(1,1)}
            
            for key, val in x2_ups.items():
                if self.scene==key:
                    x2 = MyUpSampling2D(size=(1,1), num_pixels=val)(x2)
                    break
                
            for key, val in x3_ups.items():
                if self.scene==key:
                    x3 = MyUpSampling2D(size=(1,1), num_pixels=val)(x3)
                    break
            
        # concatenate feature maps
        top = keras.layers.concatenate([x1, x2, x3], name='feature_concat')
        
        if dataset_name=='CDnet':
            if(self.scene=='wetSnow'):
                top = MyUpSampling2D(size=(1,1), num_pixels=(3,0))(top)
            elif(self.scene=='skating'):
                top = MyUpSampling2D(size=(1,1), num_pixels=(2,3))(top)
        
        # Transposed Conv
        top = self.transposedConv(top)
        
        if dataset_name=='CDnet':
            if(self.scene=='tramCrossroad_1fps'):
                top = MyUpSampling2D(size=(1,1), num_pixels=(2,0))(top)
            elif(self.scene=='bridgeEntry'):
                top = MyUpSampling2D(size=(1,1), num_pixels=(2,2))(top)
            elif(self.scene=='fluidHighway'):
                top = MyUpSampling2D(size=(1,1), num_pixels=(2,0))(top)
            elif(self.scene=='streetCornerAtNight'): 
                top = MyUpSampling2D(size=(1,1), num_pixels=(1,0))(top)
                top = Cropping2D(cropping=((0, 0),(0, 1)))(top)
            elif(self.scene=='tramStation'):  
                top = Cropping2D(cropping=((1, 0),(0, 0)))(top)
            elif(self.scene=='twoPositionPTZCam'):
                top = MyUpSampling2D(size=(1,1), num_pixels=(0,2))(top)
            elif(self.scene=='turbulence2'):
                top = Cropping2D(cropping=((1, 0),(0, 0)))(top)
                top = MyUpSampling2D(size=(1,1), num_pixels=(0,1))(top)
            elif(self.scene=='turbulence3'):
                top = MyUpSampling2D(size=(1,1), num_pixels=(2,0))(top)

        vision_model = Model(inputs=[input_1, input_2, input_3], outputs=top, name='vision_model')
        opt = keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08, decay=0.0)
        
        # Since UCSD has no void label, we do not need to filter out
        if dataset_name == 'UCSD':
            c_loss = loss2
            c_acc = acc2
        else:
            c_loss = loss
            c_acc = acc
            
        vision_model.compile(loss=c_loss, optimizer=opt, metrics=[c_acc])
        return vision_model
    
    ### FgSegNet_S
    
    def FPM(self, x):
        x1 = MaxPooling2D((2, 2), strides=(1,1), padding='same')(x)
        x1 = Conv2D(64, (1, 1), padding='same')(x1)
        
        x2 = Conv2D(64, (3, 3), padding='same')(x)
        
        x3 = Conv2D(64, (3, 3), padding='same', dilation_rate=4)(x)
        
        x4 = Conv2D(64, (3, 3), padding='same', dilation_rate=8)(x)
        
        x5 = Conv2D(64, (3, 3), padding='same', dilation_rate=16)(x)
        
        x = keras.layers.concatenate([x1, x2, x3, x4, x5], axis=-1)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout2D(0.25)(x)
        return x
        
    def initModel_S(self, dataset_name):
        assert dataset_name in ['CDnet', 'SBI', 'UCSD'], 'dataset_name must be either one in ["CDnet", "SBI", "UCSD"]]'
        assert len(self.img_shape)==3
        h, w, d = self.img_shape
        
        input_1 = Input(shape=(h, w, d), name='input')
        vgg_layer_output = self.VGG16(input_1)
        model = Model(inputs=input_1, outputs=vgg_layer_output, name='model')
        model.load_weights(self.vgg_weights_path, by_name=True)
        
        unfreeze_layers = ['block4_conv1','block4_conv2', 'block4_conv3']
        for layer in model.layers:
            if(layer.name not in unfreeze_layers):
                layer.trainable = False
                
        x = model.output
        
        if dataset_name=='CDnet':
            x1_ups = {'streetCornerAtNight':(0,1), 'tramStation':(1,0), 'turbulence2':(1,0)}
            for key, val in x1_ups.items():
                if self.scene==key:
                    # upscale by adding number of pixels to each dim.
                    x = MyUpSampling2D(size=(1,1), num_pixels=val)(x)
                    break
                
        x = self.FPM(x)
        x = self.transposedConv(x)
        
        if dataset_name=='CDnet':
            if(self.scene=='tramCrossroad_1fps'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0))(x)
            elif(self.scene=='bridgeEntry'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,2))(x)
            elif(self.scene=='fluidHighway'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0))(x)
            elif(self.scene=='streetCornerAtNight'): 
                x = MyUpSampling2D(size=(1,1), num_pixels=(1,0))(x)
                x = Cropping2D(cropping=((0, 0),(0, 1)))(x)
            elif(self.scene=='tramStation'):  
                x = Cropping2D(cropping=((1, 0),(0, 0)))(x)
            elif(self.scene=='twoPositionPTZCam'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(0,2))(x)
            elif(self.scene=='turbulence2'):
                x = Cropping2D(cropping=((1, 0),(0, 0)))(x)
                x = MyUpSampling2D(size=(1,1), num_pixels=(0,1))(x)
            elif(self.scene=='turbulence3'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0))(x)
        
        vision_model = Model(inputs=input_1, outputs=x, name='vision_model')
        opt = keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08, decay=0.)
        
        # Since UCSD has no void label, we do not need to filter out
        if dataset_name == 'UCSD':
            c_loss = loss2
            c_acc = acc2
        else:
            c_loss = loss
            c_acc = acc
            
        vision_model.compile(loss=c_loss, optimizer=opt, metrics=[c_acc])
        return vision_model