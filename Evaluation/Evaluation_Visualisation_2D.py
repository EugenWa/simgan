# basic matrix operations, dataloading
import os
import pickle
# - keras -
from keras.models import Model, load_model
CUDA_VISIBLE_DEVICES=0

# - plotting, img tools -
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import random

class Evaluation_Vis:
    def __init__(self, Evaluation_Result_Location, Modelname, Type, evaluation_loss_mode=0, use_eval_results_dir=True):
        # - file path info -
        self.Model_name = Modelname
        self.test_path = Modelname + ' Evaluation12'
        if use_eval_results_dir:
            self.test_path = Evaluation_Result_Location + Type + '/' + self.test_path


        # append useful directories
        self.chunk_eval_dir         = self.test_path + '/Validations'
        self.sample_dir             = self.test_path + '/Samples'
        self.training_history_dir   = self.test_path +  '/Training History'
        self.model_saves_dir        = self.test_path + '/Trained Models'


        # create directory structure
        os.makedirs(self.test_path,             exist_ok=True)
        os.makedirs(self.chunk_eval_dir,        exist_ok=True)
        os.makedirs(self.sample_dir,            exist_ok=True)
        os.makedirs(self.training_history_dir,  exist_ok=True)
        os.makedirs(self.model_saves_dir,       exist_ok=True)

        self.evaluation_mode = evaluation_loss_mode             # describes the type of loss used for evaluation
        self.evaluation_error_toleranz = 0.02

        # - misc data -
        self.best_evaluation_epoch = 0
        self.number_of_cases       = 5

    # --- Evaluation Routines ---
    def visualize_best_samples(self, gen_name, obj='', additional_subdirectory=''):
        save_path = self.sample_dir + '/' + gen_name
        if additional_subdirectory != '':
            save_path = save_path + '/' + additional_subdirectory

        A = np.load(save_path + '/%s%s' % (obj, 'A.npy'))
        B = np.load(save_path + '/%s%s' % (obj, 'B.npy'))
        T = np.load(save_path + '/%s%s' % (obj, 'T.npy'))
        function_choice = np.load(save_path + '/%s%s' % (obj, 'Ch.npy'))



        for i in range(A.shape[0]):
            fig = plt.figure(figsize=(12, 3))

            plt.subplot(141)
            plt.title('Input')
            plt.imshow(A[i, :, :, 0], cmap=plt.gray(), vmin=0, vmax=1)

            plt.subplot(142)
            plt.title('Gen Output')
            plt.imshow(T[i, :, :, 0], cmap=plt.gray(), vmin=0, vmax=1)

            plt.subplot(143)
            plt.title('Target and Gen out')
            plt.imshow(B[i, :, :, 0], cmap=plt.gray(), vmin=0, vmax=1)
            plt.imshow(T[i, :, :, 0], cmap=plt.jet(), vmin=0, vmax=1, alpha=0.7)


            eval_loss = np.mean(np.abs(B[i] - T[i]))
            print("Eval_loss: ", eval_loss)

            plt.subplot(144)
            plt.title('Error')
            plt.imshow(np.abs(B[i, :, :, 0] - T[i, :, :, 0]), cmap=plt.gray(), vmin=0, vmax=1)
            # if mode 1 is added:
            if self.evaluation_mode is 1:
                special_eval_error_1 =  np.abs(B[i,:, :, 0] - T[i,:, :, 0])
                special_eval_error_1[special_eval_error_1 < self.evaluation_error_toleranz] = 0
                plt.imshow(special_eval_error_1, cmap=plt.jet(), vmin=0, vmax=1, alpha=0.7)
                plt.suptitle('MAE ' + obj + ' Loss: ' + str(eval_loss)[0:7] + ' on function ' + str(function_choice[i]) + '\n' +
                             'Bar-Loss ' + str(np.mean(special_eval_error_1))[0:7])
            else:
                plt.suptitle('MAE ' + obj + ' Loss: ' + str(eval_loss)[0:7] + ' on function ' + str(function_choice[i]))

            fig.savefig(save_path + "/Sample_%s_%s.png" % (obj, i))
            plt.close(fig)

    def calculate_evaluation_error_picture(self, Target, Prediction):
        error = np.abs(Target - Prediction)
        if self.evaluation_mode is 0:
            return error
        elif self.evaluation_mode is 1:
            tollorable_error = error < self.evaluation_error_toleranz
            error[tollorable_error] = 0
            return error
        else:
            print('NO ERROR MODE')
            return np.zeros(Target.shape)

    def calculate_evaluation_error(self, Target, Prediction):
        if self.evaluation_mode is 0:
            return np.mean(np.abs(Target - Prediction))
        elif self.evaluation_mode is 1:
            return self.calculate_loss_outside_errorbars(Target, Prediction)
        else:
            print('NO ERROR MODE')
            return 0
    def calculate_loss_outside_errorbars(self, Target, Prediction):
        general_error = 0
        for channel in range(Target.shape[1]):
            error = np.abs(Target[:, channel] - Prediction[:, :, channel])
            # check where the error is inbound
            tollorable_error = error < self.evaluation_error_toleranz
            tollorable_error_len = np.sum(tollorable_error)

            error[tollorable_error] = 0     # negate tollerable error
            ### REMARK: it makes more sense to avarage across the whole board, since
            ###         the more values are capped reasonably the better the model performed in general
            ###         implying that the evaluation loss should be smaller
            intollerable_error = np.mean(error) #(error.shape[0] - tollorable_error_len) # this is not used
            general_error += intollerable_error

        return general_error/Target.shape[1]    # avrg over all channels

    def convert_tensor_2_fct(self, choice_A, trafo, choice_B):
        pic_A = choice_A[:, :, 0]
        pic   = trafo[:, :, 0]
        pic_B = choice_B[:, :, 0]
        return pic_A, pic, pic_B

    def visualize_chunk_data_test_epochs(self, gen_name, xA_test, xB_test, epochs, obj='', additional_subdirectory=''):
        for epoch in epochs:
            self.visualize_chunk_data_test(gen_name, xA_test, xB_test, epoch, obj, additional_subdirectory)
            print('Visualized Epoch: ', epoch)

    def visualize_chunk_data_test(self, gen_name, xA_test, xB_test, epoch, obj='', additional_subdirectory=''):
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

        save_path = self.chunk_eval_dir + '/' + gen_name
        if additional_subdirectory != '':
            save_path = save_path + '/' + additional_subdirectory

        if (epoch == '') and (obj == ''):
            save_path = save_path
        else:
            save_path = save_path + '/' + obj + str(epoch)
        os.makedirs(save_path, exist_ok=True)


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
            we_mae = np.abs(we_T - we_B)
            # evaluation loss figure
            w_axes_eval[i, 0].imshow(we_A, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_eval[i, 1].imshow(we_T, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_eval[i, 2].imshow(we_B, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_eval[i, 2].imshow(we_T, cmap=plt.jet(), vmin=0, vmax=1, alpha=0.7)
            w_axes_eval[i, 3].imshow(we_mae, cmap=plt.gray(), vmin=0, vmax=1)

            if self.evaluation_mode is 1:
                special_eval_error_1 = we_mae
                special_eval_error_1[special_eval_error_1 < self.evaluation_error_toleranz] = 0
                w_axes_eval[i, 3].imshow(special_eval_error_1, cmap=plt.jet(), vmin=0, vmax=1, alpha=0.7)

            w_axes_eval[i, 1].set_xlabel('MAE Loss: ' + str(worst_evals[i][1])[0:6])
            w_axes_eval[i, 0].set_ylabel('Img: ' + str(worst_evals[i][0]))


            be_A, be_T, be_B = self.convert_tensor_2_fct(xA[best_evals[i][0]], best_evals[i][2], xB[best_evals[i][0]])
            be_mae = np.abs(be_T - be_B)
            b_axes_eval[i, 0].imshow(be_A, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_eval[i, 1].imshow(be_T, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_eval[i, 2].imshow(be_B, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_eval[i, 2].imshow(be_T, cmap=plt.jet(), vmin=0, vmax=1, alpha=0.7)
            b_axes_eval[i, 3].imshow(be_mae, cmap=plt.gray(), vmin=0, vmax=1)
            if self.evaluation_mode is 1:
                special_eval_error_1 = be_mae
                special_eval_error_1[special_eval_error_1 < self.evaluation_error_toleranz] = 0
                b_axes_eval[i, 3].imshow(special_eval_error_1, cmap=plt.jet(), vmin=0, vmax=1, alpha=0.7)
            b_axes_eval[i, 1].set_xlabel('MAE Loss: ' + str(best_evals[i][1])[0:6])
            b_axes_eval[i, 0].set_ylabel('Img: ' + str(best_evals[i][0]))


        best_eval_fig.savefig(save_path + '/Best Evaluation Losses.png')
        worst_eval_fig.savefig(save_path + '/Worst Evaluation Losses.png')
        plt.close(best_eval_fig)
        plt.close(worst_eval_fig)


class VAE_visualisation(Evaluation_Vis):

    def __init__(self, Evaluation_Result_Location, modelname, evaluation_loss_mode=0):
        super().__init__(Evaluation_Result_Location, modelname, 'VAE', evaluation_loss_mode)

class GAN_visualisation(Evaluation_Vis):
    def __init__(self, Evaluation_Result_Location, modelname,evaluation_loss_mode=0):
        super().__init__(Evaluation_Result_Location, modelname, 'GAN',evaluation_loss_mode)


    def visualize_chunk_data_test(self, gen_name, xA_test, xB_test, epoch, obj='', additional_subdirectory=''):
        xA = xA_test
        xB = xB_test
        if epoch is -1:
            epoch=''
        save_path = self.chunk_eval_dir + '/' + gen_name
        if additional_subdirectory != '':
            save_path = save_path + '/' + additional_subdirectory

        if (epoch == '') and (obj == ''):
            save_path = save_path
        else:
            save_path = save_path + '/' + obj + str(epoch)
        os.makedirs(save_path, exist_ok=True)

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
            we_A, we_T, we_B = self.convert_tensor_2_fct(xA[worst_evals[i][0]], worst_evals[i][3], xB[worst_evals[i][0]])
            we_mae = np.abs(we_T - we_B)
            w_axes_eval[i, 0].imshow(we_A, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_eval[i, 1].imshow(we_T, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_eval[i, 2].imshow(we_B, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_eval[i, 2].imshow(we_T, cmap=plt.jet(), vmin=0, vmax=1, alpha=0.7)
            w_axes_eval[i, 3].imshow(we_mae, cmap=plt.gray(), vmin=0, vmax=1)

            if self.evaluation_mode is 1:
                special_eval_error_1 = we_mae
                special_eval_error_1[special_eval_error_1 < self.evaluation_error_toleranz] = 0
                w_axes_eval[i, 3].imshow(special_eval_error_1, cmap=plt.jet(), vmin=0, vmax=1, alpha=0.7)

            w_axes_eval[i, 1].set_xlabel('MAE Loss: ' + str(worst_evals[i][1])[0:6])
            w_axes_eval[i, 0].set_ylabel('FKT: ' + str(worst_evals[i][0]))

            be_A, be_T, be_B = self.convert_tensor_2_fct(xA[best_evals[i][0]], best_evals[i][3], xB[best_evals[i][0]])
            be_mae = np.abs(be_T - be_B)
            b_axes_eval[i, 0].imshow(be_A, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_eval[i, 1].imshow(be_T, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_eval[i, 2].imshow(be_B, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_eval[i, 2].imshow(be_T, cmap=plt.jet(), vmin=0, vmax=1, alpha=0.7)
            b_axes_eval[i, 3].imshow(be_mae, cmap=plt.gray(), vmin=0, vmax=1)
            if self.evaluation_mode is 1:
                special_eval_error_1 = be_mae
                special_eval_error_1[special_eval_error_1 < self.evaluation_error_toleranz] = 0
                b_axes_eval[i, 3].imshow(special_eval_error_1, cmap=plt.jet(), vmin=0, vmax=1, alpha=0.7)
            b_axes_eval[i, 1].set_xlabel('MAE Loss: ' + str(best_evals[i][1])[0:6])
            b_axes_eval[i, 0].set_ylabel('FKT: ' + str(best_evals[i][0]))

            # disc_loss figure
            bd_A, bd_T, bd_B = self.convert_tensor_2_fct(xA[best_disc_losses[i][0]], best_disc_losses[i][3], xB[best_disc_losses[i][0]])
            bd_mae = np.abs(bd_T - bd_B)
            b_axes_disc[i, 0].imshow(bd_A, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_disc[i, 1].imshow(bd_T, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_disc[i, 2].imshow(bd_B, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_disc[i, 2].imshow(bd_T, cmap=plt.jet(), vmin=0, vmax=1, alpha=0.7)
            b_axes_disc[i, 3].imshow(bd_mae, cmap=plt.gray(), vmin=0, vmax=1)

            if self.evaluation_mode is 1:
                special_eval_error_1 = bd_mae
                special_eval_error_1[special_eval_error_1 < self.evaluation_error_toleranz] = 0
                b_axes_disc[i, 3].imshow(special_eval_error_1, cmap=plt.jet(), vmin=0, vmax=1, alpha=0.7)

            b_axes_disc[i, 1].set_xlabel('MAE Loss: ' + str(best_evals[i][1])[0:6])
            b_axes_disc[i, 0].set_ylabel('FKT: ' + str(best_evals[i][0]))


            wd_A, wd_T, wd_B = self.convert_tensor_2_fct(xA[worst_disc_losses[i][0]], worst_disc_losses[i][3],
                                                       xB[worst_disc_losses[i][0]])
            wd_mae = np.abs(wd_T - wd_B)
            w_axes_disc[i, 0].imshow(wd_A, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_disc[i, 1].imshow(wd_T, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_disc[i, 2].imshow(wd_B, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_disc[i, 2].imshow(wd_T, cmap=plt.jet(), vmin=0, vmax=1, alpha=0.7)
            w_axes_disc[i, 3].imshow(wd_mae, cmap=plt.gray(), vmin=0, vmax=1)

            if self.evaluation_mode is 1:
                special_eval_error_1 = wd_mae
                special_eval_error_1[special_eval_error_1 < self.evaluation_error_toleranz] = 0
                w_axes_disc[i, 3].imshow(special_eval_error_1, cmap=plt.jet(), vmin=0, vmax=1, alpha=0.7)
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



