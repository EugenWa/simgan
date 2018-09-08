# - keras -
import keras
from keras.layers import BatchNormalization, Conv2D, Input, Conv2DTranspose, LeakyReLU, Add, Lambda, MaxPooling2D, \
    UpSampling2D, Concatenate, Dropout
from keras.models import Model, load_model
import numpy as np
CUDA_VISIBLE_DEVICES=0

# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

'''
    
'''
def build_encoder_basic(encoder_input, e_filters=[3, 9, 21, 33], use_batch_normalisation=True, use_dropout=False):
    kr_size=(3, 3)


    if use_batch_normalisation:
        encoder_input = BatchNormalization()(encoder_input)

    c_L = encoder_input
    c_L = Conv2D(e_filters[0], kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(c_L)
    c_L = LeakyReLU(0.2)(c_L)
    c_L = MaxPooling2D()(c_L)
    # create encoding blocks
    for i in range(1, len(e_filters)):
        c_L = Conv2D(e_filters[i], kernel_size=kr_size, strides=(1, 1), padding='same')(c_L)
        c_L = LeakyReLU(0.2)(c_L)
        c_L = MaxPooling2D()(c_L)

        if use_batch_normalisation and (i%2 is 0):
            c_L = BatchNormalization()(c_L)
        if use_dropout:
            c_L = Dropout(0.2)(c_L)


    enc_output = c_L#Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_L)

    return enc_output


def build_decoder_basic(decoder_input, d_filters=[33, 21, 9, 3], use_batch_normalisation=True, use_dropout=False):
    kr_size = (3, 3)
    if use_batch_normalisation:
        decoder_input = BatchNormalization()(decoder_input)

    c_L = decoder_input
    # create encoding blocks
    for i in range(len(d_filters)):
        if i >= len(d_filters)-1:       # last layer so that its symetric
            kr_size = (5, 5)
        c_L = Conv2D(d_filters[i], kernel_size=kr_size, strides=(1, 1), padding='same')(c_L)
        c_L = LeakyReLU(0.2)(c_L)
        c_L = UpSampling2D()(c_L)

        if use_batch_normalisation and (i%2 is 1) and (i < len(d_filters)-1):
            c_L = BatchNormalization()(c_L)
        if use_dropout and (i < len(d_filters)-1):
            c_L = Dropout(0.2)(c_L)

    if use_batch_normalisation:
        c_L = BatchNormalization()(c_L)

    dec_output = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_L)

    return dec_output


def build_transformation_layer_basic(transform_input, layer_amount, layer_size=33, use_batch_normalisation=True, use_dropout=False):
    block_in = transform_input
    for i in range(layer_amount):
        r_1 = Conv2D(layer_size, kernel_size=(3, 3), strides=(1, 1), padding='same')(block_in)
        r_1 = LeakyReLU(0.2)(r_1)
        r_2 = Conv2D(layer_size, kernel_size=(3, 3), strides=(1, 1), padding='same')(r_1)
        r_2 = LeakyReLU(0.2)(r_2)
        if use_batch_normalisation:
            r_2 = BatchNormalization()(r_2)
        if use_dropout and ((i%2 is 1) or (0 < i and (i is len(layer_amount)-2))):
            r_2 = Dropout(0.2)(r_2)

        # concatenate the residuum
        con_1 = Concatenate()([block_in, r_2])
        block_in = Conv2D(layer_size, kernel_size=(1, 1), strides=(1, 1), padding='same')(con_1)

    return block_in






