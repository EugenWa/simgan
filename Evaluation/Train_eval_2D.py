# basic matrix operations, data loading
import os
from shutil import copyfile
import pickle
# - keras -
import numpy as np
import random
import datetime

class Evaluation:
    def __init__(self, Evaluation_Result_Location , Modelname, Type, Model_config_path, incl_features=False, evaluation_loss_mode=0, use_eval_results_dir=True):
        # - file path info -
        self.Model_name = Modelname
        self.test_path = Modelname + ' Evaluation2D'
        if use_eval_results_dir:
            self.test_path = Evaluation_Result_Location + Type + '/' + self.test_path       # '../../Evaluation Results/'


        # append useful directories
        self.chunk_eval_dir     = self.test_path + '/Validations'
        self.sample_dir         = self.test_path + '/Samples'
        self.training_history_dir   = self.test_path + '/'  + 'Training History'
        self.model_saves_dir        = self.test_path + '/'  + 'Trained Models'


        # create directory structure
        os.makedirs(self.test_path,             exist_ok=True)
        os.makedirs(self.chunk_eval_dir,        exist_ok=True)
        os.makedirs(self.sample_dir,            exist_ok=True)
        os.makedirs(self.training_history_dir,  exist_ok=True)
        os.makedirs(self.model_saves_dir,       exist_ok=True)

        # copy config to this location
        copyfile(Model_config_path, self.test_path + '/used_config.ini')

        # Evaluation also on features
        self.include_features = incl_features

        self.evaluation_mode = evaluation_loss_mode             # describes the type of loss used for evaluation
        self.evaluation_error_toleranz = 0.005                  # hence abs it's going this much in both directions

        # - misc data -
        self.best_evaluation_epoch = 0
        self.number_of_cases       = 5


    # --- Evaluation Routines ---
    '''
        obj attribute to diversify path names if necessary, for instance if ID-mapping and Domain-mapping
        are supposed to be sampled and differentiated
    '''
    def sample_best_model_output(self, xA_test, xB_test, gen_A_B, gen_name, obj='', additional_subdirectory='', amount=10):
        function_choice = random.sample(range(xA_test.shape[0]), amount)        # so that samples are different

        # choose source samples
        if self.include_features:
            xA_s, xA_feat = xA_test
            choices_A = [xA_s[function_choice], xA_feat[function_choice]]
        else:
            choices_A   = xA_test[function_choice]

        transformed = gen_A_B(choices_A)
        choices_B   = xB_test[function_choice]

        for i in range(amount):
            eval_loss = self.calculate_evaluation_error(choices_B[i], transformed[i])
            print('MAE-', i, ' Loss: ', eval_loss)

        # save
        save_path = self.sample_dir + '/' + gen_name
        if additional_subdirectory != '':
            save_path = save_path + '/' + additional_subdirectory
        os.makedirs(save_path, exist_ok=True)

        np.save(save_path + '/%s%s' % (obj, 'A.npy'), choices_A)
        np.save(save_path + '/%s%s' % (obj, 'B.npy'), choices_B)
        np.save(save_path + '/%s%s' % (obj, 'T.npy'), transformed)
        # save function choice
        np.save(save_path + '/%s%s' % (obj, 'Ch.npy'), function_choice)



    def calculate_evaluation_error(self, Target, Prediction):
        if self.evaluation_mode is 0:
            return np.mean(np.abs(Target - Prediction))
        elif self.evaluation_mode is 1:
            return self.calculate_loss_outside_errorbars(Target, Prediction)
        else:
            print('NO ERROR MODE')
        return 0

    def calculate_loss_outside_errorbars_Trim(self, Target, Prediction):
        general_error = 0
        for channel in range(Target.shape[1]):
            error = np.abs(Target[:, :, channel] - Prediction[:, :, channel])
            # check where the error is inbound
            tollorable_error = error < self.evaluation_error_toleranz
            #tollorable_error_len = np.sum(tollorable_error)

            error[tollorable_error] = 0     # negate tollerable error
            ### REMARK: it makes more sense to avarage across the whole board, since
            ###         the more values are capped reasonably the better the model performed in general
            ###         implying that the evaluation loss should be smaller
            intollerable_error = np.mean(error) #(error.shape[0] - tollorable_error_len) # this is not used
            general_error += intollerable_error

        return general_error/Target.shape[1]    # avrg over all channels

    def calculate_loss_outside_errorbars(self, Target, Prediction):
        general_error = 0
        for channel in range(Target.shape[1]):
            error = np.abs(Target[:,:, channel] - Prediction[:, :, channel])
            # check where the error is inbound
            tollorable_error = error < self.evaluation_error_toleranz
            tollorable_error_len = np.sum(tollorable_error)

            error[tollorable_error] = 0     # negate tollerable error
            ### REMARK: it makes more sense to avarage across the whole board, since
            ###         the more values are capped reasonably the better the model performed in general
            ###         implying that the evaluation loss should be smaller
            ### THIS WILLBE DROPED, BECAUSE IT ARTIFICIALLY REDUCES THE LOSS, WITHOUT IMPROVEMENT!
            intollerable_error = np.sum(error) / (error.shape[0]*error.shape[1] - tollorable_error_len) # this is not used
            general_error += intollerable_error

        return general_error/Target.shape[1]    # avrg over all channels


    def evaluate_gen_only(self, model, xA_test, xB_test):
        """
            Evaluates the Model on Data
        :param model:       Generator to be Evaluated, has to be a SISO-system
        :param xA_test:     Source Data
        :param xB_test:     Target Data
        :return:            Validation score
        """
        if self.evaluation_mode is 0:
            return model.evaluate(xA_test, xB_test)
        elif self.evaluation_mode is 1:
            eval_losses = 0
            for i in range(xA_test.shape[0]):
                # generator prediction
                prediction = model.predict(xA_test[i][np.newaxis, :, :, :])[0, :, :, :]
                eval_losses += self.calculate_evaluation_error(xB_test[i], prediction)

            return eval_losses/xA_test.shape[0]
        else:
            print('NO LOSS')
        return 0


    def evaluate_model_on_testdata_chunk_gen_only(self, generator, gen_name, xA_test, xB_test, epoch, obj='', additional_subdirectory=''):
        """
            Evaluates (or rather validates) the passed Model on the Evaluation set
        :param generator:               Model.predict
        :param gen_name:                name, cause Model.predict was passed
        :param xA_test:
        :param xB_test:
        :param epoch:                   epoch to be validated, if -1 is passed -> epoch=''
        :param obj:                     additional string to add to path
        :param additional_subdirectory  To create a subfolder if needed
        :return:                        returns the mean absolute error
        """
        if epoch is -1:
            epoch = ''

        xA = xA_test
        xB = xB_test

        # to sum over all losses
        eval_losses = 0

        # amount of best and worst images to collect
        number_of_cases = self.number_of_cases
        best_evals  = []
        worst_evals = []

        # evaluate
        for i in range(xA.shape[0]):
            # generator prediction
            prediction = (generator(xA[i][np.newaxis, :, :, :]))[0, :, :, :]
            eval_loss = self.calculate_evaluation_error(xB[i], prediction)

            # store the 5 best and worst outcomes
            # - for the evaluation losses
            if len(worst_evals) < number_of_cases:                                      # since both lists are filled equally fast
                worst_evals.append([i, eval_loss, prediction])
                best_evals.append( [i, eval_loss, prediction])
            else:
                worst_evals_best = min(worst_evals, key = lambda t: t[1])               # sample with the lowest loss
                best_evals_worst = max(best_evals,  key = lambda t: t[1])                # sample with the highest loss
                if worst_evals_best[1] < eval_loss:
                    worst_evals[worst_evals.index(worst_evals_best)]    = [i, eval_loss, prediction]
                if best_evals_worst[1] > eval_loss:
                    best_evals[best_evals.index(best_evals_worst)]      = [i, eval_loss, prediction]

            eval_losses += eval_loss

        # average over losses to get the full picture
        mean_eval_loss = eval_losses/xA.shape[0]

        # - sort worst cases -
        worst_evals = sorted(worst_evals,   key = lambda t:t[1], reverse=True)
        best_evals  = sorted(best_evals,    key = lambda t:t[1])

        # save
        save_path = self.chunk_eval_dir + '/' + gen_name
        if additional_subdirectory != '':
            save_path = save_path + '/' + additional_subdirectory

        if (epoch == '') and (obj == ''):
            save_path = save_path
        else:
            save_path = save_path + '/' + obj + str(epoch)
        os.makedirs(save_path, exist_ok=True)

        # dump best eval
        with open(save_path + '/Best Eval', "wb") as fp:
            pickle.dump(best_evals, fp)
        # dump best eval
        with open(save_path + '/Worst Eval', "wb") as fp:
            pickle.dump(worst_evals, fp)

        # save loss
        to_save = mean_eval_loss  # its assumed that disc and generators arent mixed
        return to_save


class VAE_evaluation(Evaluation):

    def __init__(self, Evaluation_Result_Location, modelname, Model_config_path, incl_features=False, evaluation_loss_mode=0):
        super().__init__(Evaluation_Result_Location, modelname, 'VAE',  Model_config_path, incl_features, evaluation_loss_mode)

        # create fields to save data
        # # #
        self.best_loss_id = np.inf
        self.best_loss_no = np.inf
        self.best_epoch_id = 0
        self.best_epoch_no = 0
        # # #
        # loss history
        self.id_loss_epoch = []
        self.no_loss_epoch = []
        self.valid_loss_id = []
        self.valid_loss_no = []

    def dump_training_history(self):
        # create a dictionary to simplify
        loss_history_ID = dict()
        loss_history_NO = dict()

        loss_history_ID['loss']     = self.id_loss_epoch
        loss_history_ID['val_loss'] = self.valid_loss_id

        loss_history_NO['loss']     = self.no_loss_epoch
        loss_history_NO['val_loss'] = self.valid_loss_no


        # save history
        with open(self.training_history_dir + '/Hist_ID', "wb") as fp:
            pickle.dump(loss_history_ID, fp)
        with open(self.training_history_dir + '/Hist_NO', "wb") as fp:
            pickle.dump(loss_history_NO, fp)


    def save_testing_results(self, Best_ID_Model_test_loss_id, Best_ID_Model_test_loss_no, Best_NO_Model_test_loss_id, Best_NO_Model_test_loss_no):
        with open(self.test_path + '/Test Loss ID', "wb") as fp:
            pickle.dump([Best_ID_Model_test_loss_id, Best_ID_Model_test_loss_no], fp)
        with open(self.test_path + '/Test Loss NO', "wb") as fp:
            pickle.dump([Best_NO_Model_test_loss_id, Best_NO_Model_test_loss_no], fp)


class GAN_evaluation(Evaluation):
    def __init__(self, Evaluation_Result_Location,  modelname,  Model_config_path, incl_features=False,evaluation_loss_mode=0):
        super().__init__(Evaluation_Result_Location, modelname, 'GAN', Model_config_path, incl_features,evaluation_loss_mode)

        # create fields to save data
        # # #
        self.best_loss_id = np.inf
        self.best_loss_no = np.inf
        self.best_epoch_id = 0
        self.best_epoch_no = 0
        # # #
        # loss history
        self.id_loss_epoch = []
        self.no_loss_epoch = []
        self.ges_loss_epoch = []
        self.valid_loss_id = []
        self.valid_loss_no = []

        # Gan specific
        self.best_loss_id_GAN = [np.inf, np.inf]
        self.best_loss_no_GAN = [np.inf, np.inf]
        self.best_epoch_id_GAN = 0
        self.best_epoch_no_GAN = 0
        # # #
        # loss history
        self.gan_id_loss_epoch  = []
        self.gan_no_loss_epoch  = []
        self.gan_valid_loss_id  = []
        self.gan_valid_loss_no  = []

        self.gan_valid_loss_id_close2point5 = [np.inf, [np.inf, np.inf]]
        self.gan_valid_loss_no_close2point5 = [np.inf, [np.inf, np.inf]]
        self.best_epoch_id_GAN_close2point5 = 0
        self.best_epoch_no_GAN_close2point5 = 0

        self.gan_disc_loss_epoch = []
        self.gan_best_disc_loss = [np.inf]
        self.gan_best_disc_epoch = 0

    def dump_training_history_GAN(self):
        # --- save history ---
        # create saving history object
        training_history_GAN_id = dict()
        training_history_GAN_no = dict()

        training_history_GAN_id['loss']         = [g[0] for g in self.gan_id_loss_epoch]
        training_history_GAN_id['std']          = [g[1] for g in self.gan_id_loss_epoch]
        training_history_GAN_id['val_loss']     = self.gan_valid_loss_id

        with open(self.training_history_dir + '/ID_LOSS_GAN', "wb") as fp:
            pickle.dump(training_history_GAN_id, fp)

        training_history_GAN_no['loss']        = [g[0] for g in self.gan_no_loss_epoch]
        training_history_GAN_no['std']         = [g[1] for g in self.gan_no_loss_epoch]
        training_history_GAN_no['val_loss']    = self.gan_valid_loss_no

        with open(self.training_history_dir + '/NO_LOSS_GAN', "wb") as fp:
            pickle.dump(training_history_GAN_no, fp)


    def vae_save_testing_results(self, Best_ID_Model_test_loss_id, Best_ID_Model_test_loss_no, Best_NO_Model_test_loss_id, Best_NO_Model_test_loss_no):
        with open(self.test_path + '/Test Loss ID', "wb") as fp:
            pickle.dump([Best_ID_Model_test_loss_id, Best_ID_Model_test_loss_no], fp)
        with open(self.test_path + '/Test Loss NO', "wb") as fp:
            pickle.dump([Best_NO_Model_test_loss_id, Best_NO_Model_test_loss_no], fp)

    def evaluate_gan_minimalistic(self, gen_model, disc_model, xA_test, xB_test, patch_data):
        """
            Evaluates the Model on Data (A->B)
        :param gen_model:   Generator to be Evaluated, has to be a SISO-system
        :param disc_model:  Discriminator
        :param xA_test:     Source Data
        :param xB_test:     Target Data
        :param patch_data   [Number of Patches, Individual Patch Width]
        :return:            Validation score
        """
        if self.evaluation_mode is 0:
            gen_eval_score = gen_model.evaluate(xA_test, xB_test)

        elif self.evaluation_mode is 1:
            eval_losses = 0
            for i in range(xA_test.shape[0]):
                # generator prediction
                gen_prediction = gen_model.predict(xA_test[i][np.newaxis, :, :, :])[0, :, :, :]
                eval_losses += self.calculate_evaluation_error(xB_test[i], gen_prediction)

            gen_eval_score = eval_losses/xA_test.shape[0]
        else:
            print('NO LOSS')
            gen_eval_score = 0

        ae_predc = gen_model.predict(xA_test)
        # construct reasonable disc_batch
        disc_eval_labels = np.zeros((xB_test.shape[0] + ae_predc.shape[0],))
        disc_eval_labels[0:xB_test.shape[0]] = np.ones((xB_test.shape[0],))
        disc_eval_batch = np.zeros((xB_test.shape[0] + ae_predc.shape[0],) + xB_test[0].shape)
        disc_eval_batch[0:xB_test.shape[0]] = xB_test
        disc_eval_batch[xB_test.shape[0]:] = ae_predc

        disc_eval_indiv_patch_score = []
        for p_i in range(patch_data[0]):
            for p_j in range(patch_data[2]):
                disc_eval_indiv_patch_score.append(disc_model.evaluate(disc_eval_batch[:, p_i * patch_data[1]:(p_i + 1) * patch_data[1], p_j * patch_data[3]:(p_j + 1)*patch_data[3], :], disc_eval_labels))

        disc_eval_score = np.mean(disc_eval_indiv_patch_score, axis=0)

        return gen_eval_score, disc_eval_score


    def evaluate_gan_on_testdata_chunk(self, generator, gen_name, discriminator, patch_data, xA_test, xB_test, epoch, obj='', additional_subdirectory=''):
        if epoch < 0:
            epoch=''
        xA = xA_test
        xB = xB_test

        print('Started evaluation ', datetime.datetime.now())
        # to sum over all losses
        eval_losses = 0
        disc_losses = 0

        # amount of best and worst images to collect
        number_of_cases = self.number_of_cases
        best_evals = []
        worst_evals = []
        best_disc_losses = []
        worst_disc_losses =[]
        correct_disc_predictions = 0        # will flow to the accuracy evaluation metric

        # evaluate
        for i in range(xA.shape[0]):
            # generator prediction
            prediction = generator(xA[i][np.newaxis, :, :, :])[0, :, :, :]
            eval_loss = self.calculate_evaluation_error(xB[i], prediction)

            # discriminator prediction
            disc_predctions = []
            for p_i in range(patch_data[0]):
                for p_j in range(patch_data[2]):
                    disc_predctions.append(discriminator(prediction[np.newaxis, p_i*patch_data[1]:(p_i+1)*patch_data[1], p_j*patch_data[3]:(p_j+1)*patch_data[3], :]))
            disc_pred = np.mean(disc_predctions)
            disc_loss = np.mean(np.abs(0 - disc_pred))            # abstand zum correct predictetem label, gt is 0

            #print('Discriminator pred: ',np.mean(disc_pred))
            if disc_pred < 0.5:
                correct_disc_predictions += 1

            # store the 5 best and worst outcomes
            # - for the evaluation losses
            if len(worst_evals) < number_of_cases:                                      # since both lists are filled equally fast
                worst_evals.append([i, eval_loss, disc_loss, prediction])
                best_evals.append([i, eval_loss, disc_loss, prediction])
            else:
                worst_evals_best = min(worst_evals, key = lambda t: t[1])
                best_evals_worst = max(best_evals, key = lambda t: t[1])
                if worst_evals_best[1] < eval_loss:
                    worst_evals[worst_evals.index(worst_evals_best)] = [i, eval_loss, disc_loss,prediction]
                if best_evals_worst[1] > eval_loss:
                    best_evals[best_evals.index(best_evals_worst)] = [i, eval_loss, disc_loss, prediction]

            # - for the worst disc_losses
            if len(worst_disc_losses) < number_of_cases:
                worst_disc_losses.append([i, eval_loss, disc_loss, prediction])
                best_disc_losses.append([i, eval_loss, disc_loss, prediction])
            else:
                worst_evals_best = min(worst_disc_losses, key=lambda t: t[2])
                best_evals_worst = max(best_disc_losses, key=lambda t: t[2])
                if worst_evals_best[2] < disc_loss:
                    worst_disc_losses[worst_disc_losses.index(worst_evals_best)] = [i, eval_loss, disc_loss, prediction]
                if best_evals_worst[2] > eval_loss:
                    best_disc_losses[best_disc_losses.index(best_evals_worst)] = [i, eval_loss, disc_loss, prediction]


            disc_losses += disc_loss
            eval_losses += eval_loss

        # avarage over losses to get the full picture
        mean_disc_loss = disc_losses/xA.shape[0]
        mean_eval_loss = eval_losses/xA.shape[0]

        disc_accuracy = correct_disc_predictions / xA.shape[0]

        # - sort worst cases -
        worst_evals = sorted(worst_evals, key = lambda t:t[1], reverse=True)
        best_evals = sorted(best_evals, key=lambda t:t[1])
        worst_disc_losses = sorted(worst_disc_losses, key=lambda t: t[2], reverse=True)
        best_disc_losses = sorted(best_disc_losses, key=lambda t:t[2])

        # save
        save_path = self.chunk_eval_dir + '/' + gen_name
        if additional_subdirectory != '':
            save_path = save_path + '/' + additional_subdirectory

        if (epoch == '') and (obj == ''):
            save_path = save_path
        else:
            save_path = save_path + '/' + obj + str(epoch)
        os.makedirs(save_path, exist_ok=True)

        # dump best eval
        with open(save_path + '/Best Eval Gen', "wb") as fp:
            pickle.dump(best_evals, fp)
        # dump best eval
        with open(save_path + '/Worst Eval Gen', "wb") as fp:
            pickle.dump(worst_evals, fp)
        with open(save_path + '/Best Eval Disc', "wb") as fp:
            pickle.dump(best_disc_losses, fp)
            # dump best eval
        with open(save_path + '/Worst Eval Disc', "wb") as fp:
            pickle.dump(worst_disc_losses, fp)

        print('End evaluation ', datetime.datetime.now())

        # --- save average losses ---
        to_save = [mean_eval_loss, mean_disc_loss, disc_accuracy]  # its assumed that disc and generators arent mixed
        return to_save

