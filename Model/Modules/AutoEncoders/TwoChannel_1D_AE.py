import os
import sys
import pickle
import numpy as np

sys.path.insert(0, '../../../')
# - keras -
import keras
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from Model.Modules.Generators1D.Generator_Basic import build_basic_decoder, build_basic_encoder, build_basic_transformation_layers
from Model.Modules.Static_Objects import possible_loss_functions, possible_optimizers



def build_basic_AE_encoder(inp, filters, res_layers, relu_param=0.3, use_batch_norm=True, use_dropout=False):
    """
        Helper Function to build an encoder really fast
    :param inp:
    :param filters:
    :param res_layers:
    :param relu_param:
    :param use_batch_norm:
    :param use_dropout:
    :return:
    """

    feat = build_basic_encoder(inp, filters, relu_param, use_batch_norm, use_dropout)
    if res_layers > 0:
        tfeat = build_basic_transformation_layers(feat, res_layers, filters[-1], relu_param, use_batch_norm, use_dropout)
    else:
        tfeat = feat
    return tfeat


class Two_Channel_Auto_Encoder:
    def __init__(self, ae_name, img_shape, cfg, AE_eval):
        self.name = ae_name
        self.optimizers = possible_optimizers  # {'adam':keras.optimizers.adam}
        self.loss_functions = possible_loss_functions  # {'mae':keras.losses.mae, 'mse':keras.losses.mse}
        self.ae_eval = AE_eval
        self.seed = cfg['SEED']
        self.NO_fit_mode = cfg['VAE']['NO_FIT_MODE']

        ae_input_id = Input(shape=img_shape)
        ae_input_no = Input(shape=img_shape)

        lr = cfg['VAE']['LEARNING_RATE']
        lr_decay = cfg['VAE']['LR_DEF']
        relu_param = cfg['VAE']['RELU_PARAM']
        optimizer = self.optimizers[cfg['VAE']['OPTIMIZER']]
        use_drop_out = cfg['VAE']['USE_DROP_OUT']
        use_batch_normalisation = cfg['VAE']['USE_BATCH_NORM']
        trafo_layers = cfg['VAE']['TRAFO_LAYERS']
        filters = cfg['VAE']['FILTERS']
        filters_rev = list(reversed(filters))
        feature_inp = Input(shape=(img_shape[0] / (2 ** len(filters)), filters[-1]))

        # --- Build Encoder Models ---
        trafo_layers_id = 0
        encoder_ID_features = build_basic_AE_encoder(ae_input_id, filters, trafo_layers_id, relu_param, use_batch_normalisation, use_drop_out)
        encoder_NO_features = build_basic_AE_encoder(ae_input_no, filters, trafo_layers, relu_param, use_batch_normalisation, use_drop_out)

        # create Models
        self.encoder_ID = Model(ae_input_id, encoder_ID_features, name='encoder_ID')
        self.encoder_NO = Model(ae_input_no, encoder_NO_features, name='encoder_NO')

        # --- Build Decoder Model ---
        decoder_ID_out = build_basic_decoder(feature_inp, filters_rev, relu_param, use_batch_normalisation, use_drop_out)
        self.Decoder = Model(feature_inp, decoder_ID_out, name='decoder')

        # compile decoder
        self.decoder_optimizer = optimizer(lr, lr_decay)
        self.Decoder.compile(optimizer=self.decoder_optimizer, loss=cfg['VAE']['DECODER_LOSS'])

        img_loss_weight = 10
        feat_loss_weight = 1
        img_loss_type = self.loss_functions[cfg['VAE']['IMAGE_LOSS']]
        feat_loss_type = self.loss_functions[cfg['VAE']['FEATURE_LOSS']]
        # compile all Models (decoder was compiled before)
        self.ae_optimizer = optimizer(lr, lr_decay)


        # --- Build Auto-Encoder Models ---
        reconstructed_id = self.Decoder(self.encoder_ID(ae_input_id))
        self.AE_ID      = Model(inputs=[ae_input_id], outputs=[reconstructed_id], name='AE_ID')
        self.AE_ID_F    = Model(inputs=[ae_input_id], outputs=[encoder_ID_features], name='AE_ID_F')

        self.AE_ID.compile(optimizer=self.ae_optimizer, loss=img_loss_type)
        self.AE_ID_F.compile(optimizer=self.ae_optimizer, loss=feat_loss_type)  # non trainable
        # !!! freeze decoder !!!
        self.Decoder.trainable = False
        reconstructed_no = self.Decoder(self.encoder_NO(ae_input_no))
        self.AE_NO = Model(inputs=[ae_input_no], outputs=[reconstructed_no], name='AE_NO')

        # --- Build Multi channel Auto-Encoder Models ---
        # combine to a two channel model
        full_reconst_no = self.AE_NO(ae_input_no)

        self.Two_Ch_train = Model(inputs=[ae_input_id, ae_input_no], outputs=[reconstructed_id, full_reconst_no, encoder_NO_features], name='TwoCh_train')
        self.TwoCh_AE = Model(inputs=[ae_input_id, ae_input_no], outputs=[reconstructed_id, reconstructed_no], name='TwoCh')
        self.TwoCH_ID_Frozen = Model(inputs=[ae_input_no], outputs=[full_reconst_no, encoder_NO_features], name='TwoCh_IDFr')



        self.AE_NO.compile(optimizer=self.ae_optimizer, loss=img_loss_type)
        self.TwoCh_AE.compile(optimizer=self.ae_optimizer, loss=[img_loss_type, img_loss_type]) # preferably non trainable
        self.Two_Ch_train.compile(optimizer=self.ae_optimizer, loss=[img_loss_type, img_loss_type, feat_loss_type], loss_weights=[img_loss_weight, img_loss_weight, feat_loss_weight])
        self.TwoCH_ID_Frozen.compile(optimizer=self.ae_optimizer, loss=[img_loss_type, feat_loss_type])


        # --- Display Layouts ---
        print('Encoder-ID Layout: ')
        self.encoder_ID.summary()
        print('-------------------------------------------------------------------------')
        print('\nEncoder-NO Layout: ')
        self.encoder_NO.summary()
        print('-------------------------------------------------------------------------')
        print('\nDecoder Layout: ')
        self.Decoder.summary()

        print('Complete AE_Models')
        print('-------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------')
        print('\nAE_ID Layout: ')
        self.AE_ID.summary()
        print('-------------------------------------------------------------------------')
        print('\nAE_NO Layout: ')
        self.AE_NO.summary()
        print('-------------------------------------------------------------------------')
        print('\nTwoCh Layout: ')
        self.TwoCh_AE.summary()


        # plot models to files
        keras.utils.plot_model(self.AE_ID, self.ae_eval.model_saves_dir + '/' + self.AE_ID.name + '.png')
        keras.utils.plot_model(self.AE_NO, self.ae_eval.model_saves_dir + '/' + self.AE_NO.name + '.png')
        keras.utils.plot_model(self.TwoCh_AE, self.ae_eval.model_saves_dir + '/' + self.TwoCh_AE.name + '.png')
        keras.utils.plot_model(self.Two_Ch_train, self.ae_eval.model_saves_dir + '/' + self.Two_Ch_train.name + '.png')

        keras.utils.plot_model(self.encoder_ID, self.ae_eval.model_saves_dir + '/' + self.encoder_ID.name + '.png')
        keras.utils.plot_model(self.encoder_NO, self.ae_eval.model_saves_dir + '/' + self.encoder_NO.name + '.png')
        keras.utils.plot_model(self.Decoder, self.ae_eval.model_saves_dir + '/' + self.Decoder.name + '.png')


    def load_pretrained_model(self, Type):
        if Type=='ID':
            try:
                self.AE_ID.load_weights(self.ae_eval.model_saves_dir + '/AE_ID.h5')
            except Exception:
                print('NO ID MODEL')
        else:
            try:
                self.AE_NO.load_weights(self.ae_eval.model_saves_dir + '/AE_NO.h5')
            except Exception:
                print('NO NO MODEL')




    # routines so that loading a pre-trained AE is easier
    def load_AE_ID(self, path):
        self.AE_ID.load_weights(path)

    def load_AE_NO(self, path):
        self.AE_NO.load_weights(path)

    def save_AE_ID(self):
        self.AE_ID.save(self.ae_eval.model_saves_dir + '/AE_ID.h5')

    def save_AE_NO(self):
        self.AE_ID.save(self.ae_eval.model_saves_dir + '/AE_NO_ID.h5')
        self.AE_NO.save(self.ae_eval.model_saves_dir + '/AE_NO.h5')

    def load_TwoCh_AE(self, path):
        self.TwoCh_AE.load_weights(path)

    def save_model(self):
        self.Decoder.save(self.ae_eval.model_saves_dir + '/decoder.h5')
        self.encoder_ID.save(self.ae_eval.model_saves_dir + '/encoder_ID.h5')
        self.encoder_NO.save(self.ae_eval.model_saves_dir + '/encoder_NO.h5')
        self.save_AE_ID()
        self.save_AE_NO()
        self.TwoCh_AE.save(self.ae_eval.model_saves_dir + '/TwoCh.h5')

    def load_Model(self, path):
        if self.Decoder is not None:
            self.Decoder.load_weights(path + '/decoder.h5')

        if self.encoder_ID is not None:
            self.encoder_ID.load_weights(path + '/encoder_ID.h5')

        if self.encoder_NO is not None:
            self.encoder_NO.load_weights(path + '/encoder_NO.h5')

        if self.AE_ID is not None:
            self.load_AE_ID(path + '/AE_ID.h5')

        if self.AE_NO is not None:
            self.load_AE_NO(path + '/AE_NO.h5')

        if self.TwoCh_AE is not None:
            self.TwoCh_AE.load_weights(path + '/TwoCh.h5')

    def fit_NO(self, xA_train, xB_train, xA_eval, xB_eval, xA_test, xB_test, epochs, batch_size, shuffle, verbose, save_best_only):
        if self.NO_fit_mode == 1:#'NO_only':
            self.fit_NO_only(xA_train, xB_train, xA_eval, xB_eval, xA_test, xB_test, epochs, batch_size, shuffle, verbose, save_best_only)
        elif self.NO_fit_mode == 2:#'2ch_paired':
            self.fit_Two_normal(xA_train, xB_train, xA_eval, xB_eval, xA_test, xB_test, epochs, batch_size, shuffle, verbose, save_best_only)
        else:       # FEAT
            self.fit_Two_channel_feat(xA_train, xB_train, xA_eval, xB_eval, xA_test, xB_test, epochs, batch_size, shuffle, verbose, save_best_only)



    def fit_ID(self, xA_train, xB_train, xA_eval, xB_eval, xA_test, xB_test, epochs, batch_size, shuffle, verbose, save_best_only):
        np.random.seed(self.seed)  # equal shuffling
        check_pointer_ID = ModelCheckpoint(self.ae_eval.model_saves_dir + '/AE_ID.h5', verbose=verbose, save_best_only=save_best_only)
        history_id = self.AE_ID.fit(xB_train, xA_train, validation_data=(xB_eval, xA_eval), shuffle=shuffle, epochs=epochs, batch_size=batch_size, callbacks=[check_pointer_ID])

        self.load_AE_ID(self.ae_eval.model_saves_dir + '/AE_ID.h5')
        eval_score_id_best = self.AE_ID.evaluate(xB_test, xA_test, batch_size=batch_size)
        print('Best ID-Model score: ', eval_score_id_best)
        print('----------------------------------------------------------------------')

        # save histories
        try:
            with open(self.ae_eval.training_history_dir + '/Hist_ID', "rb") as fp:
                history_id_previous = pickle.load(fp)
        except Exception:
            history_id_previous = {'loss': [], 'val_loss': []}

        # append new found data
        history_id_previous['loss'].extend(history_id.history['loss'])
        history_id_previous['val_loss'].extend(history_id.history['val_loss'])

        # safe history
        with open(self.ae_eval.training_history_dir + '/Hist_ID', "wb") as fp:
            pickle.dump(history_id_previous, fp)


    def fit_NO_only(self, xA_train, xB_train, xA_eval, xB_eval, xA_test, xB_test, epochs, batch_size, shuffle, verbose, save_best_only):
        print('NO ONLY')
        np.random.seed(self.seed)  # equal shuffling
        check_pointer_NO = ModelCheckpoint(self.ae_eval.model_saves_dir + '/AE_NO.h5', verbose=verbose, save_best_only=save_best_only)
        history_no = self.AE_NO.fit(xB_train, xA_train, validation_data=(xB_eval, xA_eval), shuffle=shuffle, epochs=epochs, batch_size=batch_size, callbacks=[check_pointer_NO])

        self.load_AE_NO(self.ae_eval.model_saves_dir + '/AE_NO.h5')
        eval_score_no_best = self.AE_NO.evaluate(xB_test, xA_test, batch_size=batch_size)
        print('Best NO-Model score: ', eval_score_no_best)
        print('----------------------------------------------------------------------')

        # save histories
        try:
            with open(self.ae_eval.training_history_dir + '/Hist_NO', "rb") as fp:
                history_no_previous = pickle.load(fp)
        except Exception:
            history_no_previous = {'loss': [], 'val_loss': []}

        # append new found data
        history_no_previous['loss'].extend(history_no.history['loss'])
        history_no_previous['val_loss'].extend(history_no.history['val_loss'])

        # safe training history
        with open(self.ae_eval.training_history_dir + '/Hist_NO', "wb") as fp:
            pickle.dump(history_no_previous, fp)


    def fit_Two_normal(self, xA_train, xB_train, xA_eval, xB_eval, xA_test, xB_test, epochs, batch_size, shuffle, verbose, save_best_only):
        print('NO NORMAL')
        np.random.seed(self.seed)  # equal shuffling
        check_pointer_TwCH = ModelCheckpoint(self.ae_eval.model_saves_dir + '/TwoCh.h5', verbose=verbose, save_best_only=save_best_only)
        history_TwCH = self.TwoCh_AE.fit([xA_train, xB_train], [xA_train, xA_train], validation_data=([xA_eval, xB_eval], [xA_eval, xA_eval]), shuffle=shuffle, epochs=epochs, batch_size=batch_size, callbacks=[check_pointer_TwCH])

        # manage saves
        self.load_TwoCh_AE(self.ae_eval.model_saves_dir + '/TwoCh.h5')
        self.save_AE_ID()
        self.save_AE_NO()

        # load best
        self.load_AE_ID(self.ae_eval.model_saves_dir + '/AE_ID.h5')
        eval_score_id_best = self.AE_ID.evaluate(xA_test, xA_test, batch_size=batch_size)
        self.load_AE_NO(self.ae_eval.model_saves_dir + '/AE_NO.h5')
        eval_score_no_best = self.AE_NO.evaluate(xB_test, xA_test, batch_size=batch_size)

        print('Best ID-Model score: ', eval_score_id_best)
        print('Best NO-Model score: ', eval_score_no_best)
        print('----------------------------------------------------------------------')

        with open(self.ae_eval.training_history_dir + '/Hist_NO', "wb") as fp:
            pickle.dump(history_TwCH.history, fp)

    def train_on_batch_ID_frozen(self, xA_batch, xB_batch):
        features_id = self.AE_ID_F.predict([xA_batch])
        loss_no = self.TwoCH_ID_Frozen.train_on_batch([xB_batch], [xA_batch, features_id])
        return loss_no

    def train_on_batch_feat(self, xA_batch, xB_batch):
        features_id = self.AE_ID_F.predict([xA_batch])
        loss_no = self.Two_Ch_train.train_on_batch([xA_batch, xB_batch], [xA_batch, xA_batch, features_id])
        return loss_no


    def fit_Two_channel_feat(self, xA_train, xB_train, xA_eval, xB_eval, xA_test, xB_test, epochs, batch_size, shuffle, verbose, save_best_only):
        print('NO ONLY FEATures')
        np.random.seed(self.seed)  # equal shuffling
        number_of_iterations = int(xA_train.shape[0] / batch_size)
        equal_shuffle = []
        for i in range(epochs):
            a = np.arange(xA_train.shape[0])
            if shuffle:
                np.random.shuffle(a)
            equal_shuffle.append(a)

        hist_id = []
        hist_no = []
        validation_hist_id = []
        validation_hist_no = []
        best_validation_id = np.inf
        best_validation_no = np.inf
        for epoch in range(epochs):
            xA_train = xA_train[equal_shuffle[epoch]]
            xB_train = xB_train[equal_shuffle[epoch]]

            collect_loss_id = []  # lists to save batch losses
            collect_loss_no = []
            for batch_i in range(number_of_iterations):
                xA_batch = xA_train[batch_i * batch_size:(batch_i + 1) * batch_size]
                xB_batch = xB_train[batch_i * batch_size:(batch_i + 1) * batch_size]

                # feed into model
                loss_id = self.AE_ID.train_on_batch(xA_batch, xA_batch)
                if self.NO_fit_mode == 3:
                    loss_no = self.train_on_batch_ID_frozen(xA_batch, xB_batch)
                else:
                    loss_no = self.train_on_batch_feat(xA_batch, xB_batch)

                print(f"Epoch/Batch: {epoch:3}/{batch_i:6}    Loss ID: {str(loss_id)[0:12]:15}  Loss NO: {loss_no}")
                #print('Epoch/Batch: ', epoch, ' ', batch_i, '    Loss ID: ', loss_id, '   Loss NO:', loss_no)
                collect_loss_id.append(loss_id)
                collect_loss_no.append(loss_no)
            hist_id.append(np.mean(collect_loss_id))
            hist_no.append(np.mean(collect_loss_no))

            # evaluate model
            validation_id = self.AE_ID.evaluate([xA_eval], [xA_eval], batch_size)
            validation_no = self.TwoCh_AE.evaluate([xA_eval, xB_eval], [xA_eval, xA_eval], batch_size)

            print('Validation ID: ', validation_id)
            print('Validation Tch: ', validation_no)
            print('-------------------------------------------')
            validation_hist_id.append(validation_id)
            validation_hist_no.append(validation_no)

            if save_best_only:
                if validation_id < best_validation_id:
                    self.save_AE_ID()
                if validation_no[0] < best_validation_no:
                    self.save_AE_NO()
            else:
                if epoch >= epochs - 1:
                    self.save_AE_ID()
                    self.save_AE_NO()

        # Evaluation on test
        self.load_AE_ID(self.ae_eval.model_saves_dir + '/AE_ID.h5')
        eval_score_id_best = self.AE_ID.evaluate(xA_test, xA_test, batch_size=batch_size)
        self.load_AE_NO(self.ae_eval.model_saves_dir + '/AE_NO.h5')
        self.load_AE_ID(self.ae_eval.model_saves_dir + '/AE_NO_ID.h5')
        eval_score_no_best = self.AE_NO.evaluate(xB_test, xA_test, batch_size=batch_size)

        print('Best ID-Model score: ', eval_score_id_best)
        print('Best NO-Model score: ', eval_score_no_best)
        print('----------------------------------------------------------------------')

        # save histories
        try:
            with open(self.ae_eval.training_history_dir + '/Hist_NO', "rb") as fp:
                history_no = pickle.load(fp)
        except Exception:
            history_no = {'loss': [], 'val_loss': []}
        try:
            with open(self.ae_eval.training_history_dir + '/Hist_ID', "rb") as fp:
                history_id = pickle.load(fp)
        except Exception:
            history_id = {'loss': [], 'val_loss': []}

        # append new found data
        history_id['loss'].extend(hist_id)
        history_id['val_loss'].extend(validation_hist_id)

        history_no['loss'].extend(hist_no)
        history_no['val_loss'].extend(validation_hist_no)

        with open(self.ae_eval.training_history_dir + '/Hist_ID', "wb") as fp:
            pickle.dump(history_id, fp)

        with open(self.ae_eval.training_history_dir + '/Hist_NO', "wb") as fp:
            pickle.dump(history_no, fp)













