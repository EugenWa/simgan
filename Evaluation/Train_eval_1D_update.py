# basic matrix operations, dataloading
import os
import pickle
# - keras -
from keras.models import Model, load_model
CUDA_VISIBLE_DEVICES=0

# - plotting, img tools -
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import random

class Evaluation:
    def __init__(self, Modelname, data_config, use_eval_results_dir=True):
        # - file path info -
        self.Model_name = Modelname
        self.test_path = Modelname + '_evaluation1D'
        if use_eval_results_dir:
            self.test_path = '../../Evaluation Results/' + self.test_path


        # append useful directories
        self.chunk_eval_dir     = self.test_path + '/Validations'
        self.sample_dir         = self.test_path + '/Samples'

        # create directory structure
        os.makedirs(self.test_path,         exist_ok=True)
        os.makedirs(self.chunk_eval_dir,    exist_ok=True)
        os.makedirs(self.sample_dir,        exist_ok=True)


        # - misc data -
        self.best_evaluation_epoch = 0
        self.number_of_cases       = 5
        self.data_config = {'len':data_config[2], 'interval':data_config[1]}
        self.time_axis  =  np.arange(0, self.data_config['interval'], self.data_config['interval']/self.data_config['len'])

    # --- Evaluation Routines ---
    '''
        obj attribute to diversify path names if necessary, for instance if ID-mapping and Domain-mapping
        are supposed to be sampled and differentiated
    '''
    def sample_best_model_output(self, xA_test, xB_test, gen_A_B, gen_name, obj='', amount=10):
        function_choice = random.sample(range(xA_test.shape[0]), amount)        # so that samples are different
        #results = np.zeros((amount, ) + xA_test.shape[1:])                      # create tensor to save

        transformed = gen_A_B(xA_test[function_choice])
        choices_A   = xA_test[function_choice]
        choices_B   = xB_test[function_choice]

        for i in range(amount):
            eval_loss = np.mean(np.abs(choices_B[i] - transformed[i]))
            print('MAE-', i, ' Loss: ', eval_loss)

        # save
        save_path = self.sample_dir + '/' + gen_name
        os.makedirs(save_path, exist_ok=True)
        np.save(save_path + '/%s%s' % (obj, 'A.npy'), choices_A)
        np.save(save_path + '/%s%s' % (obj, 'B.npy'), choices_B)
        np.save(save_path + '/%s%s' % (obj, 'T.npy'), transformed)
        # save function choice
        np.save(save_path + '/%s%s' % (obj, 'Ch.npy'), function_choice)

    def visualize_best_samples(self, gen_name, obj=''):
        save_path = self.sample_dir + '/' + gen_name
        A = np.load(save_path + '/%s%s' % (obj, 'A.npy'))
        B = np.load(save_path + '/%s%s' % (obj, 'B.npy'))
        T = np.load(save_path + '/%s%s' % (obj, 'T.npy'))
        function_choice = np.load(save_path + '/%s%s' % (obj, 'Ch.npy'))

        for i in range(A.shape[0]):
            fig = plt.figure()

            plt.subplot(141)
            plt.title('Input')
            plt.plot(self.time_axis, A[i, :, 0], color='blue')

            plt.subplot(142)
            plt.title('Gen Output')
            plt.plot(self.time_axis, T[i, :, 0], color='cyan')

            plt.subplot(143)
            plt.title('Target and Gen out')
            plt.plot(self.time_axis, B[i, :, 0], color='green')
            plt.plot(self.time_axis, T[i, :, 0], color='cyan')


            plt.subplot(144)
            plt.title('Error')
            plt.plot(self.time_axis, np.abs(B[i, :, 0] - T[i, :, 0]), color='red')

            eval_loss = np.mean(np.abs(B[i] - T[i]))
            print("Eval_loss: ", eval_loss)

            plt.suptitle('MAE ' + obj + ' Loss: ' + str(eval_loss)[0:7] + ' on function ' + str(function_choice[i]))

            fig.savefig(save_path + "/Sample_%s%s.png" % (obj, i))
            plt.close(fig)



    def convert_tensor_2_fct(self, choice_A, trafo, choice_B):
        pic_A = choice_A[:, 0]
        pic   = trafo[:, 0]
        pic_B = choice_B[:, 0]
        return pic_A, pic, pic_B


    def evaluate_model_on_testdata_chunk_gen_only(self, generator, gen_name, xA_test, xB_test, epoch, obj=''):
        """
            Evaluates (or rather validates) the passed Model on the Evaluation set
        :param generator:       Model.predict
        :param gen_name:        name, cause Model.predict was passed
        :param xA_test:
        :param xB_test:
        :param epoch:           epoch to be validated, if -1 is passed -> epoch=''
        :param obj:             additional string to add to path
        :return:                returns the mean absolute error
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
            prediction = generator(xA[i][np.newaxis, :, :])[0, :, :]
            eval_loss = np.mean(np.abs(xB[i] - prediction))

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
        save_path = self.chunk_eval_dir + '/' + gen_name + '/' + obj + str(epoch)
        os.makedirs(save_path, exist_ok=True)

        # dump best eval
        with open(save_path + '/Best Eval', "wb") as fp:
            pickle.dump(best_evals, fp)
        # dump best eval
        with open(save_path + '/Worst Eval', "wb") as fp:
            pickle.dump(worst_evals, fp)

        # save loss
        to_save = mean_eval_loss  # its assumed that disc and generators arent mixed
        with open(save_path + '/Mean_loss', "wb") as fp:
            pickle.dump(to_save, fp)

        return to_save



    def visualize_chunk_data_test(self, gen_name, xA_test, xB_test, epoch, obj=''):
        """
            Visualizes the saved evaluation Data
        :param gen_name:    Name of the generator that was used
        :param xA_test:     Here, because the sample of the dataset wasnt saved; this set isnt shuffled!
        :param xB_test:     Here, because the sample of the dataset wasnt saved; this set isnt shuffled!
        :param epoch:       Epoch to be visualized, if -1 it will be relaced with ''
        :param obj:         default data
        :return:            -
        """
        if epoch is -1:
            epoch = ''

        xA = xA_test
        xB = xB_test

        save_path = self.chunk_eval_dir + '/' + gen_name + '/' + obj + str(epoch)
        # - sort worst cases -
        # dump best eval
        with open(save_path + '/Best Eval', "rb") as fp:
            best_evals = pickle.load(fp)
        # dump best eval
        with open(save_path + '/Worst Eval', "rb") as fp:
            worst_evals = pickle.load(fp)

        number_of_cases = self.number_of_cases
        cols = ['Original', 'Translated', 'Target+Trans', 'Error']
        best_eval_fig, b_axes_eval = plt.subplots(nrows=number_of_cases, ncols=4)
        plt.tight_layout()
        worst_eval_fig,w_axes_eval = plt.subplots(nrows=number_of_cases, ncols=4)
        plt.tight_layout()


        for ax, col in zip(b_axes_eval[0], cols):
            ax.set_title(col)
        for ax, col in zip(w_axes_eval[0], cols):
            ax.set_title(col)

        for i in range(number_of_cases):
            we_A, we_T, we_B = self.convert_tensor_2_fct(xA[worst_evals[i][0]],worst_evals[i][2], xB[worst_evals[i][0]])
            # evaluation loss figure
            w_axes_eval[i, 0].plot(self.time_axis, we_A, color='blue')
            w_axes_eval[i, 1].plot(self.time_axis, we_T, color='cyan')
            w_axes_eval[i, 2].plot(self.time_axis, we_B, color='green')
            w_axes_eval[i, 2].plot(self.time_axis, we_T, color='cyan')
            w_axes_eval[i, 3].plot(self.time_axis, (np.abs(we_T - we_B)), color='red')
            w_axes_eval[i, 1].set_xlabel('MAE Loss: ' + str(worst_evals[i][1])[0:6])
            w_axes_eval[i, 0].set_ylabel('Img: ' + str(worst_evals[i][0]))


            be_A, be_T, be_B = self.convert_tensor_2_fct(xA[best_evals[i][0]], best_evals[i][2], xB[best_evals[i][0]])
            b_axes_eval[i, 0].plot(self.time_axis, be_A, color='blue')
            b_axes_eval[i, 1].plot(self.time_axis, be_T, color='cyan')
            b_axes_eval[i, 2].plot(self.time_axis, be_B, color='green')
            b_axes_eval[i, 2].plot(self.time_axis, be_T, color='cyan')
            b_axes_eval[i, 3].plot(self.time_axis, (np.abs(be_T - be_B)), color='red')
            b_axes_eval[i, 1].set_xlabel('MAE Loss: ' + str(best_evals[i][1])[0:6])
            b_axes_eval[i, 0].set_ylabel('Img: ' + str(best_evals[i][0]))

        best_eval_fig.savefig(save_path + '/Best Evaluation Losses.png')
        worst_eval_fig.savefig(save_path + '/Worst Evaluation Losses.png')
        plt.close(best_eval_fig)
        plt.close(worst_eval_fig)


class VAE_evaluation(Evaluation):

    def __init__(self, modelname, cfg):
        super().__init__(modelname + '_VAE', cfg)
        self.training_history_dir   = self.test_path + '/' + self.Model_name + ' Training History'
        self.model_saves_dir        = self.test_path + '/' + self.Model_name + ' Trained Models'
        os.makedirs(self.training_history_dir, exist_ok=True)
        os.makedirs(self.model_saves_dir, exist_ok=True)

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

    def dump_training_history(self):
        # save history
        with open(self.training_history_dir + '/ID_LOSS', "wb") as fp:
            pickle.dump(self.id_loss_epoch, fp)
        with open(self.training_history_dir + '/NO_LOSS', "wb") as fp:
            pickle.dump(self.no_loss_epoch, fp)
        with open(self.training_history_dir + '/GES_LOSS', "wb") as fp:
            pickle.dump(self.ges_loss_epoch, fp)

        # save validation history
        with open(self.training_history_dir + '/VAL_ID_LOSS', "wb") as fp:
            pickle.dump(self.valid_loss_id, fp)
        with open(self.training_history_dir + '/VAL_NO_LOSS', "wb") as fp:
            pickle.dump(self.valid_loss_no, fp)

    def q_save_model_NO(self, validation_loss_no_epoch, epoch):
        if self.best_loss_no > validation_loss_no_epoch:
            self.best_loss_no = validation_loss_no_epoch
            self.best_epoch_no = epoch
            return True
        return False

    def q_save_model_ID(self, validation_loss_id_epoch, epoch):
        if self.best_loss_id > validation_loss_id_epoch:
            self.best_loss_id = validation_loss_id_epoch
            self.best_epoch_id = epoch
            return True
        return False

    def switch_best_epoch(self, key=True):
        if key:
            self.best_evaluation_epoch = self.best_epoch_id
        else:
            self.best_evaluation_epoch = self.best_epoch_no

    def save_testing_results(self, Best_ID_Model_test_loss_id, Best_ID_Model_test_loss_no, Best_NO_Model_test_loss_id, Best_NO_Model_test_loss_no):
        with open(self.test_path + '/Test Loss ID', "wb") as fp:
            pickle.dump([Best_ID_Model_test_loss_id, Best_ID_Model_test_loss_no], fp)
        with open(self.test_path + '/Test Loss NO', "wb") as fp:
            pickle.dump([Best_NO_Model_test_loss_id, Best_NO_Model_test_loss_no], fp)



class Generator_evaluation(Evaluation):

    def __init__(self, modelname):
        super().__init__(modelname + 'Gen')

        # - Loss history -
        self.complete_loss = []  # includes the generators -> add sufficent labeling
        self.complete_loss.append([])  # first list for first epoch
        self.avrg_complete_losses = []

        # - best model preservation -
        self.best_evaluation_epoch = 0
        self.best_genloss = np.inf
        self.best_Model = None  # will consist of: [Modeltype, gen_los, ....]

        self.equal_shuffle_among_models = True

    def add_epoch_loss(self):
        # calculate avarage loss of the last epoch
        avrg_loss = np.mean([g[0] for g in self.complete_loss[-1]])
        self.avrg_complete_losses.append(avrg_loss)
        self.complete_loss.append([])
        return avrg_loss

    def add_batch_loss(self, batchloss, epoch_i, batch_i):
        self.complete_loss[len(self.complete_loss) - 1].append([batchloss, epoch_i, batch_i])

        # --- to file saving routines ---

    def load_best_model_from_Model(self, model):
        tmp = model.name
        del model
        return load_model(self.test_path + '/' + tmp + '.h5')

    def load_Model(self, model_name):
        return load_model(self.test_path + '/' + model_name + '.h5')

    def save_model(self, model):
        model.save(self.test_path + '/' + model.name + '.h5')

    def save_best_epoch_loss(self):
        to_save = [self.best_evaluation_epoch, self.best_genloss]
        with open(self.test_path + '/best_loss_epoch', "wb") as fp:
            pickle.dump(to_save, fp)




class GAN_evaluation(Evaluation):
    def __init__(self, modelname, cfg):
        super().__init__(modelname + '_GAN', cfg)
        self.training_history_dir   = self.test_path + '/' + self.Model_name + ' Training History'
        self.model_saves_dir        = self.test_path + '/' + self.Model_name + ' Trained Models'
        os.makedirs(self.training_history_dir, exist_ok=True)
        os.makedirs(self.model_saves_dir, exist_ok=True)

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
        self.gan_loss_epoch     = []
        self.gan_ges_loss_epoch = []
        self.gan_valid_loss_id  = []
        self.gan_valid_loss_no  = []

        self.gan_disc_loss_epoch = []
        self.gan_best_disc_loss = [np.inf]
        self.gan_best_disc_epoch = 0

    def dump_training_history_GAN(self):
        # --- save history ---
        # - save loss id -
        print(self.gan_id_loss_epoch)
        with open(self.training_history_dir + '/ID_LOSS_GAN_ges', "wb") as fp:
            pickle.dump([g[0][0] for g in self.gan_id_loss_epoch], fp)
        with open(self.training_history_dir + '/ID_LOSS_GAN_gen', "wb") as fp:
            pickle.dump([g[0][1] for g in self.gan_id_loss_epoch], fp)
        with open(self.training_history_dir + '/ID_LOSS_GAN_disc', "wb") as fp:
            pickle.dump([g[0][2] for g in self.gan_id_loss_epoch], fp)
        # std
        with open(self.training_history_dir + '/ID_LOSS_GAN_STD', "wb") as fp:
            pickle.dump([g[1] for g in self.gan_id_loss_epoch], fp)

        # - save loss_ges -
        with open(self.training_history_dir + '/GES_LOSS_GAN_ges', "wb") as fp:
            pickle.dump([g[0][0] for g in self.gan_ges_loss_epoch], fp)
        with open(self.training_history_dir + '/GES_LOSS_GAN_gen', "wb") as fp:
            pickle.dump([g[0][1] for g in self.gan_ges_loss_epoch], fp)
        with open(self.training_history_dir + '/GES_LOSS_GAN_features', "wb") as fp:
            pickle.dump([g[0][2] for g in self.gan_ges_loss_epoch], fp)
        with open(self.training_history_dir + '/GES_LOSS_GAN_disc', "wb") as fp:
            pickle.dump([g[0][3] for g in self.gan_ges_loss_epoch], fp)
        # std
        with open(self.training_history_dir + '/GES_LOSS_GAN_STD', "wb") as fp:
            pickle.dump([g[1] for g in self.gan_ges_loss_epoch], fp)

        # save validation history
        with open(self.training_history_dir + '/VAL_ID_LOSS_GAN_gen', "wb") as fp:
            pickle.dump([g[0] for g in self.gan_valid_loss_id], fp)
        with open(self.training_history_dir + '/VAL_ID_LOSS_GAN_disc', "wb") as fp:
            pickle.dump([g[1] for g in self.gan_valid_loss_id], fp)
        with open(self.training_history_dir + '/VAL_ID_LOSS_GAN_discAcc', "wb") as fp:
            pickle.dump([g[2] for g in self.gan_valid_loss_id], fp)
        with open(self.training_history_dir + '/VAL_NO_LOSS_GAN_gen', "wb") as fp:
            pickle.dump([g[0] for g in self.gan_valid_loss_no], fp)
        with open(self.training_history_dir + '/VAL_NO_LOSS_GAN_disc', "wb") as fp:
            pickle.dump([g[1] for g in self.gan_valid_loss_no], fp)
        with open(self.training_history_dir + '/VAL_NO_LOSS_GAN_discAcc', "wb") as fp:
            pickle.dump([g[2] for g in self.gan_valid_loss_no], fp)

    def dump_training_history_ID(self):
        # save history
        with open(self.training_history_dir + '/ID_LOSS_ID', "wb") as fp:
            pickle.dump(self.id_loss_epoch, fp)
        with open(self.training_history_dir + '/NO_LOSS_ID', "wb") as fp:
            pickle.dump(self.no_loss_epoch, fp)
        with open(self.training_history_dir + '/GES_LOSS_ID', "wb") as fp:
            pickle.dump(self.ges_loss_epoch, fp)

        # save validation history
        with open(self.training_history_dir + '/VAL_ID_LOSS_ID', "wb") as fp:
            pickle.dump(self.valid_loss_id, fp)
        with open(self.training_history_dir + '/VAL_NO_LOSS_ID', "wb") as fp:
            pickle.dump(self.valid_loss_no, fp)

    def q_vae_save_model_NO(self, validation_loss_no_epoch, epoch):
        if self.best_loss_no > validation_loss_no_epoch:
            self.best_loss_no = validation_loss_no_epoch
            self.best_epoch_no = epoch
            return True
        return False

    def q_vae_save_model_ID(self, validation_loss_id_epoch, epoch):
        if self.best_loss_id > validation_loss_id_epoch:
            self.best_loss_id = validation_loss_id_epoch
            self.best_epoch_id = epoch
            return True
        return False

    def switch_best_epoch_vae(self, key=True):
        if key:
            self.best_evaluation_epoch = self.best_epoch_id
        else:
            self.best_evaluation_epoch = self.best_epoch_no

    def vae_save_testing_results(self, Best_ID_Model_test_loss_id, Best_ID_Model_test_loss_no, Best_NO_Model_test_loss_id, Best_NO_Model_test_loss_no):
        with open(self.test_path + '/Test Loss ID', "wb") as fp:
            pickle.dump([Best_ID_Model_test_loss_id, Best_ID_Model_test_loss_no], fp)
        with open(self.test_path + '/Test Loss NO', "wb") as fp:
            pickle.dump([Best_NO_Model_test_loss_id, Best_NO_Model_test_loss_no], fp)



    def evaluate_gan_on_testdata_chunk(self, generator, gen_name, discriminator, patch_data, xA_test, xB_test, epoch, obj=''):
        if epoch < 0:
            epoch=''
        xA = xA_test
        xB = xB_test

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
            prediction = generator(xA[i][np.newaxis, :, :])[0, :, :]
            eval_loss = np.mean(np.abs(xB[i] - prediction))

            # discriminator prediction
            disc_predctions = []
            for p_i in range(patch_data[0]):
                disc_predctions.append(discriminator(prediction[np.newaxis, p_i*patch_data[1]:(p_i+1)*patch_data[1], :]))
            disc_pred = np.mean(disc_predctions)
            disc_loss = np.mean(np.square(np.ones((1,)) - disc_pred))            # abstand zum correct predictetem label

            #print('Discriminator pred: ',np.mean(disc_pred))
            if np.mean(disc_pred) < 0.5:
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
        save_path = self.chunk_eval_dir + '/' + gen_name + '/' + obj + str(epoch)
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

        # --- save average losses ---
        to_save = [mean_eval_loss, mean_disc_loss, disc_accuracy]  # its assumed that disc and generators arent mixed
        with open(save_path + 'GAN_Mean_losses' + obj, "wb") as fp:
            pickle.dump(to_save, fp)

        return to_save

    def visualize_chunk_data_test(self, gen_name, xA_test, xB_test, epoch, obj=''):
        xA = xA_test
        xB = xB_test

        save_path = self.chunk_eval_dir + '/' + gen_name + '/' + obj + str(epoch)

        # load data
        # dump best eval
        with open(save_path + '/Best Eval Gen', "rb") as fp:
            best_evals = pickle.load(fp)
        # dump best eval
        with open(save_path + '/Worst Eval Gen', "rb") as fp:
            worst_evals = pickle.load(fp)
        with open(save_path + '/Best Eval Disc', "rb") as fp:
            best_disc_losses = pickle.load(fp)
            # dump best eval
        with open(save_path + '/Worst Eval Disc', "rb") as fp:
            worst_disc_losses = pickle.load(fp)

        number_of_cases = self.number_of_cases
        cols = ['Original', 'Translated', 'Target+T', 'Error']
        best_eval_fig, b_axes_eval = plt.subplots(nrows=number_of_cases, ncols=4)
        plt.tight_layout()
        worst_eval_fig, w_axes_eval = plt.subplots(nrows=number_of_cases, ncols=4)
        plt.tight_layout()
        best_disc_fig, b_axes_disc = plt.subplots(nrows=number_of_cases, ncols=4)
        plt.tight_layout()
        worst_disc_fig, w_axes_disc = plt.subplots(nrows=number_of_cases, ncols=4)
        plt.tight_layout()


        for ax, col in zip(b_axes_eval[0], cols):
            ax.set_title(col)
        for ax, col in zip(w_axes_eval[0], cols):
            ax.set_title(col)
        for ax, col in zip(b_axes_disc[0], cols):
            ax.set_title(col)
        for ax, col in zip(w_axes_disc[0], cols):
            ax.set_title(col)

        for i in range(number_of_cases):
            we_A, we_T, we_B = self.convert_tensor_2_fct(xA[worst_evals[i][0]],worst_evals[i][3], xB[worst_evals[i][0]])
            w_axes_eval[i, 0].plot(self.time_axis, we_A, color='blue')
            w_axes_eval[i, 1].plot(self.time_axis, we_T, color='cyan')
            w_axes_eval[i, 2].plot(self.time_axis, we_B, color='green')
            w_axes_eval[i, 2].plot(self.time_axis, we_T, color='cyan')
            w_axes_eval[i, 3].plot(self.time_axis, (np.abs(we_T - we_B)), color='red')
            w_axes_eval[i, 1].set_xlabel('MAE Loss: ' + str(worst_evals[i][1])[0:6])
            w_axes_eval[i, 0].set_ylabel('FKT: ' + str(worst_evals[i][0]))

            be_A, be_T, be_B = self.convert_tensor_2_fct(xA[best_evals[i][0]], best_evals[i][3], xB[best_evals[i][0]])
            b_axes_eval[i, 0].plot(self.time_axis, be_A, color='blue')
            b_axes_eval[i, 1].plot(self.time_axis, be_T, color='cyan')
            b_axes_eval[i, 2].plot(self.time_axis, be_B, color='green')
            b_axes_eval[i, 2].plot(self.time_axis, be_T, color='cyan')
            b_axes_eval[i, 3].plot(self.time_axis, (np.abs(be_T - be_B)), color='red')
            b_axes_eval[i, 1].set_xlabel('MAE Loss: ' + str(best_evals[i][1])[0:6])
            b_axes_eval[i, 0].set_ylabel('FKT: ' + str(best_evals[i][0]))

            # disc_loss figure
            bd_A, bd_T, bd_B = self.convert_tensor_2_fct(xA[best_disc_losses[i][0]], best_disc_losses[i][3], xB[best_disc_losses[i][0]])
            b_axes_disc[i, 0].plot(self.time_axis, bd_A, color='blue')
            b_axes_disc[i, 1].plot(self.time_axis, bd_T, color='cyan')
            b_axes_disc[i, 2].plot(self.time_axis, bd_B, color='green')
            b_axes_disc[i, 2].plot(self.time_axis, bd_T, color='cyan')
            b_axes_disc[i, 3].plot(self.time_axis, (np.abs(bd_T - bd_B)), color='red')
            b_axes_disc[i, 1].set_xlabel('MAE Loss: ' + str(best_evals[i][1])[0:6])
            b_axes_disc[i, 0].set_ylabel('FKT: ' + str(best_evals[i][0]))

            wd_A, wd_T, wd_B = self.convert_tensor_2_fct(xA[worst_disc_losses[i][0]], worst_disc_losses[i][3],
                                                       xB[worst_disc_losses[i][0]])
            w_axes_disc[i, 0].plot(self.time_axis, wd_A, color='blue')
            w_axes_disc[i, 1].plot(self.time_axis, wd_T, color='cyan')
            w_axes_disc[i, 2].plot(self.time_axis, wd_B, color='green')
            w_axes_disc[i, 2].plot(self.time_axis, wd_T, color='cyan')
            w_axes_disc[i, 3].plot(self.time_axis, (np.abs(wd_T - wd_B)), color='red')
            w_axes_disc[i, 1].set_xlabel('MAE Loss: ' + str(best_evals[i][1])[0:6])
            w_axes_disc[i, 0].set_ylabel('FKT: ' + str(best_evals[i][0]))


        best_eval_fig.savefig(save_path + '/Best Evaluation Losses.png')
        worst_eval_fig.savefig(save_path + '/Worst Evaluation Losses.png')
        best_disc_fig.savefig(save_path + '/Best Discriminator Losses.png')
        worst_disc_fig.savefig(save_path + '/Worst Discriminator Losses.png')

        plt.close(best_eval_fig)
        plt.close(worst_eval_fig)
        plt.close(best_disc_fig)
        plt.close(worst_disc_fig)



