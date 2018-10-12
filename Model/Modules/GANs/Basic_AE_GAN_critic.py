# import default path
import sys
sys.path.insert(0, '../../../')

from Model.Modules.Discriminators1D.Disc_1D             import discriminator_build_4conv_oneD
from Model.Modules.Static_Objects                       import possible_loss_functions, possible_optimizers

from Model.Modules.AutoEncoders.Basic_1D_AutoEncoders                   import Basic_nRES_AE
from Model.Modules.AutoEncoders.Basic_1D_Residual_AutoEncoders          import Basic_nRES_ResidualAE
from Model.Modules.AutoEncoders.TwoChannel_1D_AutoEncoders              import Two_Channel_Auto_Encoder

# - keras -
from keras.models import Model
from keras.layers import Input, Lambda, Average
import numpy as np

import keras


class Basic_AE_Gan_Critic:
    def __init__(self, gan_name, img_shape, cfg_gan, cfg_ae, Gan_eval, AE_eval):
        self.name = gan_name
        self.optimizers         = possible_optimizers  # {'adam':keras.optimizers.adam ......}
        self.loss_functions     = possible_loss_functions  # {'mae':keras.losses.mae, 'mse':keras.losses.mse ......}
        self.gan_eval = Gan_eval
        self.seed = cfg_gan['SEED']
        self.clip_value = 0.01

        # create patch_gan
        self.patch_length_width  = cfg_gan['DISC']['PATCH_NUMBER_W']
        disc_reluparam      = cfg_gan['DISC']['RELU_PARAM']
        disc_use_drop_out   = cfg_gan['DISC']['USE_DROP_OUT']
        disc_use_batch_norm = cfg_gan['DISC']['USE_BATCH_NORM']
        disc_lr             = cfg_gan['DISC']['LEARNING_RATE']
        disc_lr_decay       = cfg_gan['DISC']['LR_DEF']

        self.patch_width = int(img_shape[0] / self.patch_length_width)

        self.gan_name = cfg_gan['MODEL_NAME']
        self.ae_name  = cfg_gan['FULL_MODEL']['VAE_NAME']
        self.disc_name= cfg_gan['FULL_MODEL']['DISC_NAME']
        model_identification_num = cfg_ae['VAE']['MODEL_ID']

        self.unsupervised = cfg_gan['FULL_MODEL']['UNSUPERVISED']



        # --- Build (V)-AE ---
        Models = [Basic_nRES_AE, Basic_nRES_ResidualAE]
        self.ae_eval = AE_eval
        self.AE_Model = Models[model_identification_num](self.ae_name, img_shape, cfg_ae, self.ae_eval)



        # --- Build Discriminator ---
        disc_input  = Input(shape=(self.patch_width, img_shape[1]))
        disc_output = discriminator_build_4conv_oneD(disc_input, disc_reluparam, disc_use_batch_norm, disc_use_drop_out)

        self.disc_optimizer = self.optimizers[cfg_gan['DISC']['OPTIMIZER']](disc_lr, disc_lr_decay)
        self.discriminator = Model(disc_input, disc_output, name=self.disc_name)

        def wasserstein_loss(y_true, y_pred):
            return keras.backend.mean(y_true * y_pred)

        self.discriminator.compile(optimizer=self.disc_optimizer, loss=wasserstein_loss, metrics=['accuracy'])
        self.discriminator.trainable = False        # freeze discriminator



        # --- Build Full Model ---
        full_model_inp = Input(shape=img_shape)
        trafo = self.AE_Model.Call()(full_model_inp)
        disc_patch_eval = []
        for p_i in range(self.patch_length_width):
            disc_patch_eval.append(self.discriminator(Lambda(lambda x: x[:, p_i * self.patch_width:(p_i + 1) * self.patch_width, :])(trafo)))
        if len(disc_patch_eval) is 1:
            disc_eval = disc_patch_eval[0]
        else:
            disc_eval = Average()(disc_patch_eval)


        self.GAN_optimizer = self.optimizers[cfg_gan['FULL_MODEL']['OPTIMIZER']](cfg_gan['FULL_MODEL']['LEARNING_RATE'], cfg_gan['FULL_MODEL']['LR_DEF'])

        if self.unsupervised:
            self.GAN = Model(inputs=[full_model_inp], outputs=[disc_eval], name=gan_name)
            self.GAN.compile(optimizer=self.GAN_optimizer, loss=wasserstein_loss, loss_weights=cfg_gan['FULL_MODEL']['LOSS_WEIGHTS'])
        else:
            self.GAN = Model(inputs=[full_model_inp], outputs=[trafo, disc_eval], name=gan_name)
            self.GAN.compile(optimizer=self.GAN_optimizer, loss=[cfg_gan['FULL_MODEL']['IMG_LOSS'], wasserstein_loss], loss_weights=cfg_gan['FULL_MODEL']['LOSS_WEIGHTS'])




        print('Full Model: ')
        self.GAN.summary()
        print('\n-------------------------------------------------------------------------------')
        # compile
        print('Discriminator')
        self.discriminator.summary()
        print('\n-------------------------------------------------------------------------------')

        keras.utils.plot_model(self.GAN, self.gan_eval.model_saves_dir + '/' + self.GAN.name + '.png')
        keras.utils.plot_model(self.discriminator, self.gan_eval.model_saves_dir + '/' + self.discriminator.name + '.png')


    def save(self, additional_obj):
        self.discriminator.save(self.gan_eval.model_saves_dir + '/Discriminator_' + additional_obj + '.h5')
        self.AE_Model.Call().save(self.gan_eval.model_saves_dir + '/AE_' + additional_obj + '.h5')
        #self.GAN.save(self.gan_eval.model_saves_dir + '/GAN_' + additional_obj + '.h5')

    def load(self, additional_obj):
        self.discriminator.load_weights(self.gan_eval.model_saves_dir + '/Discriminator_' + additional_obj + '.h5')
        self.AE_Model.Call().load_weights(self.gan_eval.model_saves_dir + '/AE_' + additional_obj + '.h5')
        #self.GAN.load_weights(self.gan_eval.model_saves_dir + '/GAN_' + additional_obj + '.h5')

    def clip_weights(self):
        # Clip critic weights
        for l in self.discriminator.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]

            l.set_weights(weights)



