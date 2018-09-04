# - keras -
import keras
from keras.layers import BatchNormalization, Conv2D, Input, Conv2DTranspose, LeakyReLU, Add, Lambda, MaxPooling2D, \
    UpSampling2D, Concatenate, Dropout
from keras.models import Model, load_model
import numpy as np
CUDA_VISIBLE_DEVICES=0
from Generator_Blocks import build_transformation_layer_basic, build_decoder_basic, build_encoder_basic





class Basic_VAE:
    def __init__(self, vae_name, img_shape, lr_options=[keras.optimizers.adam, 0.002, 0.5], path=None):
        self.name = vae_name

        if path is None:
            vae_input_id = Input(shape=img_shape)
            vae_input_no = Input(shape=img_shape)

            filters =[3, 9, 21, 33]
            filters_rev = list(reversed(filters))
            feature_inp = Input(shape=(img_shape[0]/(2**len(filters)), img_shape[1]/(2**len(filters)), filters[-1]))

            # -- VAE Model ID ---
            encoder_ID_features = build_encoder_basic(vae_input_id, filters, True, True)
            decoder_ID_out = build_decoder_basic(feature_inp, filters_rev, True, True)
            self.decoder_optimizer = lr_options[0](lr_options[1], lr_options[2])
            self.Decoder = Model(inputs=[feature_inp], outputs=[decoder_ID_out], name=vae_name + '_decoder')
            self.Decoder.summary()
            self.Decoder.compile(optimizer=self.decoder_optimizer, loss='mae')

            # build vae_id encoder
            vae__out_id = self.Decoder(encoder_ID_features)
            self.vae_id_optimizer = lr_options[0](lr_options[1], lr_options[2])
            self.VAE_ID = Model(inputs=[vae_input_id], outputs=[vae__out_id], name=vae_name + '_vaeID')
            self.VAE_ID_MultiOut = Model(inputs=[vae_input_id], outputs=[vae__out_id, encoder_ID_features], name=vae_name + '_vaeIDMOU')
            self.VAE_ID.summary()
            self.VAE_ID_MultiOut.compile(optimizer=self.vae_id_optimizer, loss=['mae', None], loss_weights=[1, None])
            self.VAE_ID.compile(optimizer=self.vae_id_optimizer, loss=['mae'])

            # freeze the decoder in the domain A->B setting hence it should learn to reconstruct the features
            self.Decoder.trainable=False

            # transformation Model
            encoder_NO_features = build_encoder_basic(vae_input_no, filters, True, True)
            transfo_NO_features = build_transformation_layer_basic(encoder_NO_features, 2, filters[-1], True, True)
            vae__out_no = self.Decoder(transfo_NO_features)
            self.vae_no_optimizer = lr_options[0](lr_options[1], lr_options[2])
            self.VAE_NO = Model(inputs=[vae_input_no], outputs=[vae__out_no, transfo_NO_features], name=vae_name + '_vaeNO')
            self.VAE_NO_MulitOut = Model(inputs=[vae_input_no], outputs=[vae__out_no, transfo_NO_features, encoder_NO_features],
                                name=vae_name + '_vaeNOMOU')
            self.VAE_NO.summary()
            self.VAE_NO.compile(optimizer=self.vae_no_optimizer, loss=['mae', 'mse'], loss_weights=[1, 1])
            self.VAE_NO_MulitOut.compile(optimizer=self.vae_no_optimizer, loss=['mae', 'mse', None], loss_weights=[1, 1, None])
        else:
            self.load_Model(path)


    def train_model_on_batch(self, xA_Batch, xB_Batch, ID_ONLY=False):
        loss_id = self.VAE_ID.train_on_batch([xB_Batch], [xB_Batch])
        if ID_ONLY:
            return loss_id, loss_id, None
        # predict features
        _, features = self.VAE_ID_MultiOut.predict(xB_Batch)
        loss_map = self.VAE_NO.train_on_batch([xA_Batch], [xB_Batch, features])
        return np.add(loss_id, loss_map)*0.5, loss_id, loss_map

    def predict_ID(self, batch):
        return self.VAE_ID_MultiOut.predict(batch)

    def predict_NO(self, batch):
        return self.VAE_NO_MulitOut.predict(batch)

    def predict_ID_img_only(self, batch):
        return self.VAE_ID.predict(batch)

    def predict_NO_img_only(self, batch):
        img, _ = self.VAE_NO.predict(batch)
        return img

    def save_model(self, path, obj=''):
        self.Decoder.save(path + '/decoder' + obj + '.h5')
        self.VAE_ID.save(path + '/vaeID' + obj+ '.h5')
        self.VAE_NO.save(path + '/vaeNO' + obj+ '.h5')
        self.VAE_ID_MultiOut.save(path + '/vaeIDMOU'+ obj + '.h5')
        self.VAE_NO_MulitOut.save(path + '/vaeNOMOU'+ obj + '.h5')

    def load_Model(self, path, obj=''):
        if self.Decoder is not None:
            del self.Decoder
        if self.VAE_ID is not None:
            del self.VAE_ID
        if self.VAE_NO is not None:
            del self.VAE_NO
        if self.VAE_NO_MulitOut is not None:
            del self.VAE_NO_MulitOut
        if self.VAE_ID_MultiOut is not None:
            del self.VAE_ID_MultiOut

        #keras.backend.clear_session()

        self.Decoder = load_model(path + '/decoder' + obj+ '.h5')
        self.VAE_ID = load_model(path + '/vaeID'+ obj+ '.h5')
        self.VAE_NO = load_model(path + '/vaeNO'+ obj+ '.h5')
        self.VAE_ID_MultiOut = load_model(path + '/vaeIDMOU'+ obj+ '.h5')
        self.VAE_NO_MulitOut = load_model(path + '/vaeNOMOU'+ obj+ '.h5')

