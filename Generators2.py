# - keras -
import keras
from keras.layers import BatchNormalization, Conv2D, Input, Conv2DTranspose, LeakyReLU, Add, Lambda, MaxPooling2D, \
    UpSampling2D, Concatenate
from keras.models import Model


def scaling(x):
    return x / 255


def upscaling(x):
    return x * 255


def scaling_shape(input_shape):
    return input_shape


Scaling = Lambda(scaling, output_shape=scaling_shape, name='Scaling')
UpScaling = Lambda(upscaling, output_shape=scaling_shape, name='UpScaling')
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

'''
    This type of Generator also holds on to the idea that the original, perfect, simulated and
    noisefree image is a valid feature representation of the Data
    So the image gets upsampled first, to some sort of 'loose' grid where information can be put 
    in from within the network - just like some form of super resolution - and after that
    compressed again to its original shape 

    Note:   The basic architecture will be 2 Convolution layers followed by and Upsampling layer each
            After those there should be a resnet type architecture build in which will serve as some
            sort of memory
            finished up by two convolutions followed by Maxpooling each
'''


def build_generator_upscale_basic(generator_input, generator_name, use_batch_normalisation=True):
    scaled_input = generator_input  # Scaling(generator_input)
    if use_batch_normalisation:
        scaled_input = BatchNormalization()(scaled_input)

    # Upsample the input
    c_1 = Conv2D(3, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(scaled_input)
    c_1 = LeakyReLU(0.2)(c_1)
    c_1 = UpSampling2D()(c_1)

    c_2 = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_1)
    c_2 = LeakyReLU(0.2)(c_2)
    c_2 = UpSampling2D()(c_2)

    # Transformation blocks
    c_3 = Conv2D(18, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_2)
    c_3 = LeakyReLU(0.2)(c_3)
    if use_batch_normalisation:
        c_3 = BatchNormalization()(c_3)

    r_1 = Conv2D(18, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_3)
    r_1 = LeakyReLU(0.2)(r_1)

    r_2 = Conv2D(18, kernel_size=(3, 3), strides=(1, 1), padding='same')(r_1)
    r_2 = LeakyReLU(0.2)(r_2)

    if use_batch_normalisation:
        r_2 = BatchNormalization()(r_2)

    con_1 = Concatenate()([c_3, r_2])
    c_4 = Conv2D(18, kernel_size=(1, 1), strides=(1, 1), padding='same')(con_1)
    c_4 = LeakyReLU(0.2)(c_4)

    c_5 = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_4)
    c_5 = LeakyReLU(0.2)(c_5)

    c_6 = Conv2D(3, kernel_size=(5, 5), strides=(1, 1), padding='same')(c_5)
    c_6 = LeakyReLU(0.2)(c_6)

    gen_output = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(c_6)
    ##gen_output = UpScaling(gen_output)

    gen_Model = Model(inputs=[generator_input], outputs=[gen_output], name=generator_name)
    gen_Model.summary()

    return gen_Model


'''
    This Adds another Resnet layer
'''


def build_generator_upscale_mtpl_resblocks(generator_input, generator_name, number_of_Resblocks=3,
                                           use_batch_normalisation=True):
    scaled_input = generator_input  # Scaling(generator_input)
    if use_batch_normalisation:
        scaled_input = BatchNormalization()(scaled_input)

    # Upsample the input
    c_1 = Conv2D(3, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(scaled_input)
    c_1 = LeakyReLU(0.2)(c_1)
    c_1 = UpSampling2D()(c_1)

    c_2 = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_1)
    c_2 = LeakyReLU(0.2)(c_2)
    c_2 = UpSampling2D()(c_2)

    # Transformation blocks
    c_3 = Conv2D(18, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_2)
    c_3 = LeakyReLU(0.2)(c_3)
    if use_batch_normalisation:
        c_3 = BatchNormalization()(c_3)

    for i in range(number_of_Resblocks):
        r_1 = Conv2D(18, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_3)
        r_1 = LeakyReLU(0.2)(r_1)

        r_2 = Conv2D(18, kernel_size=(3, 3), strides=(1, 1), padding='same')(r_1)
        r_2 = LeakyReLU(0.2)(r_2)

        if use_batch_normalisation:
            r_2 = BatchNormalization()(r_2)

        con_1 = Concatenate()([c_3, r_2])
        c_4 = Conv2D(18, kernel_size=(1, 1), strides=(1, 1), padding='same')(con_1)
        c_3 = LeakyReLU(0.2)(c_4)  # overwrites 4 to 3 to connect the loop

    c_5 = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_3)
    c_5 = LeakyReLU(0.2)(c_5)

    c_6 = Conv2D(3, kernel_size=(5, 5), strides=(1, 1), padding='same')(c_5)
    c_6 = LeakyReLU(0.2)(c_6)

    gen_output = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(c_6)
    ##gen_output = UpScaling(gen_output)

    gen_Model = Model(inputs=[generator_input], outputs=[gen_output], name=generator_name)
    gen_Model.summary()

    return gen_Model

