import os

# - keras -
import keras
from keras.layers import Conv2D, Input, Conv2DTranspose, LeakyReLU, Add, Lambda, MaxPooling2D, UpSampling2D, Concatenate
from keras.models import Model

# - plotting, img tools -
from matplotlib import pyplot as plt


def scaling(x):
    return x/255
def upscaling(x):
    return x*255

def scaling_shape(input_shape):
    return input_shape

Scaling = Lambda(scaling, output_shape=scaling_shape, name='Scaling')
UpScaling = Lambda(upscaling, output_shape=scaling_shape, name='UpScaling')

def build_generator_first(generator_input, generaotr_name):
    # convolve downwards to feature vector, use resnet_blocks to transform feature_vector to other domain, then deconvolve
    c_1 = Conv2D(16, (5, 5), padding='same', activation='relu')(generator_input)
    c_2 = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(c_1)
    c_2 = LeakyReLU(alpha=0.2)(c_2)
    c_3 = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(c_2)
    c_3 = LeakyReLU(alpha=0.2)(c_3)

    # build in a resnetblock for the transformation; 2 conv lavers + 1 residuum
    rl_1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(c_3)
    rl_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(rl_1)
    rl_2 = LeakyReLU(alpha=0.2)(rl_2)

    transformed_features = Add()([c_3, rl_2])

    d_1 = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2))(
        transformed_features)  # output_shape=c_2.shape)(transformed_features)
    d_2 = Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), output_shape=c_1.shape)(d_1)
    generator_output = Conv2DTranspose(1, 1, 1, output_shape=generator_input.shape)(d_2)

    genmod = Model(inputs=generator_input, outputs=generator_output, name=generaotr_name)
    print('Generator: ', generaotr_name, ' Summary:')
    genmod.summary()

    return genmod  # Model(inputs=generator_input, outputs=generator_output, name=generaotr_name)

def build_refiner_type_generator(generator_input, generaotr_name):
    c_1 = Conv2D(4, (3, 3), padding='same', use_bias=False)(generator_input)
    c_2 = Conv2D(16, (3, 3), padding='same')(c_1)
    c_2 = LeakyReLU(alpha=0.2)(c_2)
    c_2_p = MaxPooling2D()(c_2)

    c_3  = Conv2D(32, (3, 3), padding='same')(c_2_p)
    c_3 = LeakyReLU(alpha=0.2)(c_3)
    c_3_p = MaxPooling2D()(c_3)

    c_4 = Conv2D(32, (3, 3), padding='same')(c_3_p)
    c_4 = LeakyReLU(alpha=0.2)(c_4)
    c_4_p = MaxPooling2D()(c_4)

    c_5 = Conv2D(64, (3, 3), padding='same')(c_4_p)
    c_5 = LeakyReLU(alpha=0.2)(c_5)
    c_5_p = MaxPooling2D()(c_5)

    # resnet blocks 2*
    r_1 = Conv2D(64, (1, 1), padding='same')(c_5_p)
    r_1 = LeakyReLU(alpha=0.2)(r_1)
    r_2 = Conv2D(64, (3, 3), padding='same')(r_1)
    r_2 = LeakyReLU(alpha=0.2)(r_2)
    # concatenate both residuals
    r_3 = Concatenate()([c_5_p, r_2])
    r_3 = Conv2D(64, (1, 1), padding='same')(r_3)

    r_3 = LeakyReLU(alpha=0.2)(r_3)
    r_4 = Conv2D(64, (1, 1), padding='same')(r_3)
    r_4 = LeakyReLU(alpha=0.2)(r_4)
    r_4 = Conv2D(64, (1, 1), padding='same')(r_4)
    r_4 = LeakyReLU(alpha=0.2)(r_4)
    r_5 = Concatenate()([r_3, r_4])
    r_5 = Conv2D(64, (1, 1), padding='same')(r_5)

    # upsample
    d_5 = UpSampling2D()(r_5)
    d_5 = Conv2D(64, (3,3), padding='same', activation='relu')(d_5)

    d_4 = UpSampling2D()(d_5)
    d_4 = Conv2D(32, (3, 3), padding='same', activation='relu')(d_4)

    d_3 = UpSampling2D()(d_4)
    d_3 = Conv2D(16, (3, 3), padding='same', activation='relu')(d_3)

    d_2 = UpSampling2D()(d_3)
    d_2 = Conv2D(4, (3, 3), padding='same', activation='relu')(d_2)


    gen_output = Conv2D(1, (1, 1), padding='same', activation='tanh')(d_2)

    genmod = Model(inputs=generator_input, outputs=gen_output, name=generaotr_name)
    print('Generator: ', generaotr_name, ' Summary:')
    genmod.summary()

    return genmod


def build_generator_full_res(generator_input, generaotr_name):
    # convolve downwards to feature vector, use resnet_blocks to transform feature_vector to other domain, then deconvolve
    #scaled_inp = Scaling(generator_input)
    c_1 = Conv2D(3, (3, 3), padding='same')(generator_input)#(scaled_inp)

    c_1 = LeakyReLU(alpha=0.2)(c_1)
    #c_2 = MaxPooling2D(pool_size=(2, 2))(c_1)

    c_2 = Conv2D(8, kernel_size=(2, 2), strides=(2, 2), padding='same')(c_1)
    c_2 = LeakyReLU(alpha=0.2)(c_2)

    c_3 = Conv2D(16, kernel_size=(2, 2), strides=(1, 1), padding='same')(c_2)
    c_3 = LeakyReLU(alpha=0.2)(c_3)
    c_3 = MaxPooling2D()(c_3)

    # build up in residual form
    rl_1 = Conv2D(16, kernel_size=(4, 4), strides=(1, 1), padding='same')(c_3)
    transformed_features = Add()([c_3, rl_1])

    d_1 = Conv2DTranspose(8, kernel_size=(2, 2), strides=(2, 2))(transformed_features)
    d_1 = Add()([d_1, c_2])

    d_2 = Conv2DTranspose(3, kernel_size=(2, 2), strides=(2, 2))(d_1)
    d_2 = Add()([d_2, c_1])


    generator_output = Conv2DTranspose(1, kernel_size=(1, 1), output_shape=generator_input.shape)(d_2)
    generator_output= UpScaling(generator_output)


    genmod = Model(inputs=generator_input, outputs=generator_output, name=generaotr_name)
    print('Generator: ', generaotr_name, ' Summary:')
    genmod.summary()

    #exit()

    return genmod  # Model(inputs=generator_input, outputs=generator_output, name=generaotr_name)


def build_generator_full_res_nobias_first(generator_input, generaotr_name):
    # convolve downwards to feature vector, use resnet_blocks to transform feature_vector to other domain, then deconvolve
    #scaled_inp = Scaling(generator_input)
    c_1 = Conv2D(3, (3, 3), padding='same', use_bias=False)(generator_input)#(scaled_inp)

    c_1 = LeakyReLU(alpha=0.2)(c_1)
    #c_2 = MaxPooling2D(pool_size=(2, 2))(c_1)

    c_2 = Conv2D(8, kernel_size=(2, 2), strides=(2, 2), padding='same')(c_1)
    c_2 = LeakyReLU(alpha=0.2)(c_2)

    c_3 = Conv2D(16, kernel_size=(2, 2), strides=(1, 1), padding='same')(c_2)
    c_3 = LeakyReLU(alpha=0.2)(c_3)
    c_3 = MaxPooling2D()(c_3)

    # build up in residual form
    rl_1 = Conv2D(16, kernel_size=(4, 4), strides=(1, 1), padding='same')(c_3)
    transformed_features = Add()([c_3, rl_1])

    d_1 = Conv2DTranspose(8, kernel_size=(2, 2), strides=(2, 2))(transformed_features)
    d_1 = Add()([d_1, c_2])

    d_2 = Conv2DTranspose(3, kernel_size=(2, 2), strides=(2, 2))(d_1)
    d_2 = Add()([d_2, c_1])


    generator_output = Conv2DTranspose(1, kernel_size=(1, 1), output_shape=generator_input.shape)(d_2)
    generator_output= UpScaling(generator_output)


    genmod = Model(inputs=generator_input, outputs=generator_output, name=generaotr_name)
    print('Generator: ', generaotr_name, ' Summary:')
    genmod.summary()

    #exit()

    return genmod  # Model(inputs=generator_input, outputs=generator_output, name=generaotr_name)