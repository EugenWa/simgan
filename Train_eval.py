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

class Generator_evaluation:

    def __init__(self, Modelname):
        # - file path info -
        self.Model_name = Modelname
        self.test_path = Modelname + '_evaluation'
        os.makedirs(self.test_path, exist_ok=True)


        # - Loss history -
        self.complete_loss = []         # includes the generators -> add sufficent labeling
        self.complete_loss.append([])   # first list for first epoch
        self.avrg_complete_losses = []

        # - best model preservation -
        self.best_evaluation_epoch = 0
        self.best_genloss  = np.inf
        self.best_Model = None          # will consist of: [Modeltype, gen_los, ....]

        self.equal_shuffle_among_models = True


    def add_epoch_loss(self):
        # calculate avarage loss of the last epoch
        avrg_loss = np.mean([g[0] for g in self.complete_loss[len(self.complete_loss) - 1]])
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


    # --- Evaluation Routines ---
    def sample_evaluation(self, xA_test, xB_test, gen_A_B, epoch, obj=''):
        # sample random images from testset
        image_choice = np.random.randint(0, xA_test.shape[0])
        # notice the testset is paired, which allows for better comparison
        img_A = xA_test[image_choice]
        img_B = xB_test[image_choice]
        # convert the image into tensors whose shape can be fed into the network
        choice_A = img_A[np.newaxis, :, :, :]
        choice_B = img_B[np.newaxis, :, :, :]

        fig = plt.figure()

        plt.subplot(131)
        plt.title('Input')
        plt.axis('off')
        plt.imshow(choice_A[0, :, :, 0], cmap=plt.gray(), vmin=0, vmax=1)

        # transform the domain of the input
        trafo = gen_A_B(choice_A)

        # sclae trafo:
        t_min = np.min(trafo)
        t_max = np.max(trafo)
        t_dist = t_max - t_min
        trafo = (trafo + t_min) * (1.0 / t_dist)

        plt.subplot(132)
        plt.title('Gen Output')
        plt.axis('off')
        plt.imshow(trafo[0, :, :, 0], cmap=plt.gray(), vmin=0, vmax=1)

        plt.subplot(133)
        plt.title('Target')
        plt.axis('off')
        plt.imshow(choice_B[0, :, :, 0], cmap=plt.gray(), vmin=0, vmax=1)

        eval_loss = np.mean(np.abs(choice_B - trafo))
        print("Eval_loss: ", eval_loss)

        plt.suptitle('MAE '+ obj + ' Loss: ' + str(eval_loss)[0:7] + ' on image ' + str(image_choice))

        fig.savefig(self.test_path + "/%s_%s%s.png" % (self.Model_name, obj, epoch))
        plt.close(fig)

        return choice_A[0, :, :, 0], trafo[0, :, :, 0], choice_B[0, :, :, 0]

    '''
        obj attribute to diversify path names if necessary, for instance if ID-mapping and Domain-mapping
        are supposed to be sampled and differentiated
    '''
    def sample_best_model_output(self, xA_test, xB_test, gen_A_B, gen_name, obj='', amount=10):
        path = self.test_path + '/' + 'best_eval_epoch' + obj + str(self.best_evaluation_epoch)
        os.makedirs(path, exist_ok=True)
        image_choice = random.sample(range(xA_test.shape[0]), amount)       # so that samples are different
        for i in range(amount):
            # sample random images from testset

            # notice that the testset is paired, which allows comparison
            img_A = xA_test[image_choice[i]]
            img_B = xB_test[image_choice[i]]
            # convert the image into tensors whose shape can be fed into the network
            choice_A = img_A[np.newaxis, :, :, :]
            choice_B = img_B[np.newaxis, :, :, :]

            fig = plt.figure()

            plt.subplot(131)
            plt.title('Input')
            plt.axis('off')
            plt.imshow(choice_A[0, :, :, 0], cmap=plt.gray(), vmin=0, vmax=1)

            # transform the domain of the input
            trafo = gen_A_B(choice_A)

            # sclae trafo:
            t_min = np.min(trafo)
            t_max = np.max(trafo)
            t_dist = t_max - t_min
            trafo = (trafo + t_min) * (1.0/t_dist)

            plt.subplot(132)
            plt.title('Gen Output')
            plt.axis('off')
            plt.imshow(trafo[0, :, :, 0],    cmap=plt.gray(), vmin=0, vmax=1)

            plt.subplot(133)
            plt.title('Target')
            plt.axis('off')
            plt.imshow(choice_B[0, :, :, 0], cmap=plt.gray(), vmin=0, vmax=1)


            eval_loss = np.mean(np.abs(choice_B - trafo))
            print("Eval_loss: ", eval_loss)


            plt.suptitle('MAE ' + obj + ' Loss: ' + str(eval_loss)[0:7] + ' on image ' + str(image_choice[i]))

            fig.savefig(path + "/%s_%s%s.png" % (gen_name, obj, i))
            plt.close(fig)


    def convert_tensor2pic(self, choice_A, trafo, choice_B):
        pic_A = choice_A[:, :, 0]
        # sclae trafo:
        t_min = np.min(trafo)
        t_max = np.max(trafo)
        t_dist = t_max - t_min
        trafo = (trafo + t_min) * (1.0 / t_dist)
        pic = trafo[0, :, :, 0]
        pic_B = choice_B[:, :, 0]
        return pic_A, pic, pic_B


    def evaluate_model_on_testdata_chunk_gen_only(self, generator, gen_name, xA_test, xB_test, epoch, obj='', full_test_data=True, chunk_split=0.3, shuffle=True):
        if full_test_data:
            xA = xA_test
            xB = xB_test
        else:
            chunk_size = int(xA_test.shape[0] * chunk_split)
            imgs_start = np.random.randint(0, xA_test.shape[0] - chunk_size)

            # shuffle data if necessary
            if shuffle:
                shfl = np.arange(xA_test.shape[0])
                np.random.shuffle(shfl)
                xA_sh = xA_test[shfl]
                xB_sh = xB_test[shfl]
            else:
                xA_sh = xA_test
                xB_sh = xB_test

            # read out the samples
            xA = xA_sh[imgs_start:imgs_start+chunk_size, :, :, :]
            xB = xB_sh[imgs_start:imgs_start + chunk_size, :, :, :]

        # to sum over all losses
        eval_losses = 0

        # amount of best and worst images to collect
        number_of_cases = 5
        best_evals = []
        worst_evals = []

        # evaluate
        for i in range(xA.shape[0]):
            # generator prediction
            prediction = generator(xA[i][np.newaxis, :, :, :])
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
                    worst_evals[worst_evals.index(worst_evals_best)] = [i, eval_loss, prediction]
                if best_evals_worst[1] > eval_loss:
                    best_evals[best_evals.index(best_evals_worst)] = [i, eval_loss, prediction]

            eval_losses += eval_loss

        # average over losses to get the full picture
        mean_eval_loss = eval_losses/xA.shape[0]

        '''
        # decide whether this is the best evaluation so far
        if self.best_genloss > mean_eval_loss:
            self.best_genloss = mean_eval_loss
            self.best_evaluation_epoch = epoch
            self.save_model(generator)
            self.save_best_epoch_loss()
        '''

        # - sort worst cases -
        worst_evals = sorted(worst_evals, key = lambda t:t[1], reverse=True)
        best_evals  = sorted(best_evals, key=lambda t:t[1])

        cols = ['Original', 'Translated', 'Target']
        best_eval_fig, b_axes_eval = plt.subplots(nrows=number_of_cases, ncols=3)
        plt.tight_layout()
        worst_eval_fig,w_axes_eval = plt.subplots(nrows=number_of_cases, ncols=3)
        plt.tight_layout()


        for ax, col in zip(b_axes_eval[0], cols):
            ax.set_title(col)
        for ax, col in zip(w_axes_eval[0], cols):
            ax.set_title(col)

        for i in range(number_of_cases):
            we_A, we_T, we_B = self.convert_tensor2pic(xA[worst_evals[i][0]],worst_evals[i][2], xB[worst_evals[i][0]])
            # evaluation loss figure
            w_axes_eval[i, 0].imshow(we_A, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_eval[i, 1].imshow(we_T, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_eval[i, 2].imshow(we_B, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_eval[i, 1].set_xlabel('MAE Loss: ' + str(worst_evals[i][1])[0:6])
            w_axes_eval[i, 0].set_ylabel('Img: ' + str(worst_evals[i][0]))
            w_axes_eval[i, 0].set_yticklabels([])
            w_axes_eval[i, 0].set_xticklabels([])
            w_axes_eval[i, 1].set_yticklabels([])
            w_axes_eval[i, 1].set_xticklabels([])
            w_axes_eval[i, 0].set_xticks([])
            w_axes_eval[i, 0].set_yticks([])
            w_axes_eval[i, 1].set_xticks([])
            w_axes_eval[i, 1].set_yticks([])
            w_axes_eval[i, 2].axis('off')


            be_A, be_T, be_B = self.convert_tensor2pic(xA[best_evals[i][0]], best_evals[i][2], xB[best_evals[i][0]])
            b_axes_eval[i, 0].imshow(be_A, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_eval[i, 1].imshow(be_T, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_eval[i, 2].imshow(be_B, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_eval[i, 1].set_xlabel('MAE Loss: ' + str(best_evals[i][1])[0:6])
            b_axes_eval[i, 0].set_ylabel('Img: ' + str(best_evals[i][0]))
            b_axes_eval[i, 0].set_yticklabels([])
            b_axes_eval[i, 0].set_xticklabels([])
            b_axes_eval[i, 1].set_yticklabels([])
            b_axes_eval[i, 1].set_xticklabels([])
            b_axes_eval[i, 0].set_xticks([])
            b_axes_eval[i, 0].set_yticks([])
            b_axes_eval[i, 1].set_xticks([])
            b_axes_eval[i, 1].set_yticks([])
            b_axes_eval[i, 2].axis('off')


        path = self.test_path + '/' + obj + str(epoch)
        os.makedirs(path, exist_ok=True)

        # --- save figures ---
        best_eval_fig.savefig(path + '/' + gen_name + ' Best Evaluation Losses.png')
        worst_eval_fig.savefig(path + '/' + gen_name + ' Worst Evaluation Losses.png')
        plt.close(best_eval_fig)
        plt.close(worst_eval_fig)

        # --- save average losses ---
        to_save = mean_eval_loss       # its assumed that disc and generators arent mixed
        with open(path + '/' + gen_name + '_Mean_loss', "wb") as fp:
            pickle.dump(to_save, fp)

        return to_save

class GAN_evaluation(Generator_evaluation):
    def __init__(self, modelname):
        super().__init__(modelname + 'GAN')

    def evaluate_gan_on_testdata_chunk_gen_only(self, generator, gen_name, discriminator, patch_data, xA_test, xB_test, epoch, obj='', full_test_data=True, chunk_split=0.3, shuffle=True):
        if full_test_data:
            xA = xA_test
            xB = xB_test
        else:
            chunk_size = int(xA_test.shape[0] * chunk_split)
            imgs_start = np.random.randint(0, xA_test.shape[0] - chunk_size)

            # shuffle data if necessary
            if shuffle:
                shfl = np.arange(xA_test.shape[0])
                np.random.shuffle(shfl)
                xA_sh = xA_test[shfl]
                xB_sh = xB_test[shfl]
            else:
                xA_sh = xA_test
                xB_sh = xB_test

            # read out the samples
            xA = xA_sh[imgs_start:imgs_start+chunk_size, :, :, :]
            xB = xB_sh[imgs_start:imgs_start + chunk_size, :, :, :]

        # to sum over all losses
        eval_losses = 0
        disc_losses = 0

        # amount of best and worst images to collect
        number_of_cases = 5
        best_evals = []
        worst_evals = []
        best_disc_losses = []
        worst_disc_losses =[]
        correct_disc_predictions = 0        # will flow to the accuracy evaluation metric

        # evaluate
        for i in range(xA.shape[0]):
            # generator prediction
            prediction = generator(xA[i][np.newaxis, :, :, :])
            eval_loss = np.mean(np.abs(xB[i] - prediction))

            # discriminator prediction
            disc_predctions = []
            for p_i in range(patch_data[0]):
                for p_j in range(patch_data[1]):
                    disc_predctions.append(discriminator(prediction[:, p_i*patch_data[2]:(p_i+1)*patch_data[2], p_j*patch_data[3]:(p_j+1)*patch_data[3], :]))
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


        ##########################################
        #
        # ADD SAVE BEST MODEL incl loss and epoch where found
        #


        # - sort worst cases -
        worst_evals = sorted(worst_evals, key = lambda t:t[1], reverse=True)
        best_evals = sorted(best_evals, key=lambda t:t[1])
        worst_disc_losses = sorted(worst_disc_losses, key=lambda t: t[2], reverse=True)
        best_disc_losses = sorted(best_disc_losses, key=lambda t:t[2])

        cols = ['Original', 'Translated', 'Target']
        best_eval_fig, b_axes_eval = plt.subplots(nrows=number_of_cases, ncols=3)
        plt.tight_layout()
        worst_eval_fig,w_axes_eval = plt.subplots(nrows=number_of_cases, ncols=3)
        plt.tight_layout()
        best_disc_fig, b_axes_disc = plt.subplots(nrows=number_of_cases, ncols=3)
        plt.tight_layout()
        worst_disc_fig,w_axes_disc = plt.subplots(nrows=number_of_cases, ncols=3)
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
            we_A, we_T, we_B = self.convert_tensor2pic(xA[worst_evals[i][0]],worst_evals[i][3], xB[worst_evals[i][0]])
            # evaluation loss figure
            w_axes_eval[i, 0].imshow(we_A, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_eval[i, 1].imshow(we_T, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_eval[i, 2].imshow(we_B, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_eval[i, 1].set_xlabel('MAE Loss: ' + str(worst_evals[i][1])[0:6])
            w_axes_eval[i, 0].set_ylabel('Img: ' + str(worst_evals[i][0]))
            w_axes_eval[i, 0].set_yticklabels([])
            w_axes_eval[i, 0].set_xticklabels([])
            w_axes_eval[i, 1].set_yticklabels([])
            w_axes_eval[i, 1].set_xticklabels([])
            w_axes_eval[i, 0].set_xticks([])
            w_axes_eval[i, 0].set_yticks([])
            w_axes_eval[i, 1].set_xticks([])
            w_axes_eval[i, 1].set_yticks([])
            w_axes_eval[i, 2].axis('off')

            be_A, be_T, be_B = self.convert_tensor2pic(xA[best_evals[i][0]], best_evals[i][3], xB[best_evals[i][0]])
            b_axes_eval[i, 0].imshow(be_A, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_eval[i, 1].imshow(be_T, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_eval[i, 2].imshow(be_B, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_eval[i, 1].set_xlabel('MAE Loss: ' + str(best_evals[i][1])[0:6])
            b_axes_eval[i, 0].set_ylabel('Img: ' + str(best_evals[i][0]))
            b_axes_eval[i, 0].set_yticklabels([])
            b_axes_eval[i, 0].set_xticklabels([])
            b_axes_eval[i, 1].set_yticklabels([])
            b_axes_eval[i, 1].set_xticklabels([])
            b_axes_eval[i, 0].set_xticks([])
            b_axes_eval[i, 0].set_yticks([])
            b_axes_eval[i, 1].set_xticks([])
            b_axes_eval[i, 1].set_yticks([])
            b_axes_eval[i, 2].axis('off')

            # disc_loss figure
            bd_A, bd_T, bd_B = self.convert_tensor2pic(xA[best_disc_losses[i][0]], best_disc_losses[i][3], xB[best_disc_losses[i][0]])
            b_axes_disc[i, 0].imshow(bd_A, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_disc[i, 1].imshow(bd_T, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_disc[i, 2].imshow(bd_B, cmap=plt.gray(), vmin=0, vmax=1)
            b_axes_disc[i, 1].set_xlabel('MAE Loss: ' + str(best_evals[i][2])[0:6])
            b_axes_disc[i, 0].set_ylabel('Img: ' + str(best_evals[i][0]))
            b_axes_disc[i, 0].set_yticklabels([])
            b_axes_disc[i, 0].set_xticklabels([])
            b_axes_disc[i, 1].set_yticklabels([])
            b_axes_disc[i, 1].set_xticklabels([])
            b_axes_disc[i, 0].set_xticks([])
            b_axes_disc[i, 0].set_yticks([])
            b_axes_disc[i, 1].set_xticks([])
            b_axes_disc[i, 1].set_yticks([])
            b_axes_disc[i, 2].axis('off')

            wd_A, wd_T, wd_B = self.convert_tensor2pic(xA[worst_disc_losses[i][0]], worst_disc_losses[i][3],
                                                       xB[worst_disc_losses[i][0]])
            w_axes_disc[i, 0].imshow(wd_A, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_disc[i, 1].imshow(wd_T, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_disc[i, 2].imshow(wd_B, cmap=plt.gray(), vmin=0, vmax=1)
            w_axes_disc[i, 1].set_xlabel('MAE Loss: ' + str(worst_evals[i][1])[0:6])
            w_axes_disc[i, 0].set_ylabel('Img: ' + str(worst_evals[i][0]))
            w_axes_disc[i, 0].set_yticklabels([])
            w_axes_disc[i, 0].set_xticklabels([])
            w_axes_disc[i, 1].set_yticklabels([])
            w_axes_disc[i, 1].set_xticklabels([])
            w_axes_disc[i, 0].set_xticks([])
            w_axes_disc[i, 0].set_yticks([])
            w_axes_disc[i, 1].set_xticks([])
            w_axes_disc[i, 1].set_yticks([])
            w_axes_disc[i, 2].axis('off')

        '''
        eval_fig.tight_layout()
        disc_fig.tight_layout()
        bothbad_fig.tight_layout()
        '''

        path = self.test_path + '/' + str(epoch)
        os.makedirs(path, exist_ok=True)
        # --- save figures ---

        best_eval_fig.savefig(path + '/' + gen_name +obj + ' Best Evaluation Losses.png')
        worst_eval_fig.savefig(path + '/' + gen_name +obj+ ' Worst Evaluation Losses.png')
        best_disc_fig.savefig(path + '/' + gen_name +obj+ ' Best Discriminator Losses.png')
        worst_disc_fig.savefig(path + '/' + gen_name +obj+ ' Worst Discriminator Losses.png')

        plt.close(best_eval_fig)
        plt.close(worst_eval_fig)
        plt.close(best_disc_fig)
        plt.close(worst_disc_fig)

        # --- save average losses ---
        to_save = [mean_eval_loss, mean_disc_loss, disc_accuracy]       # its assumed that disc and generators arent mixed
        with open(path + '/' + gen_name + 'Mean_losses' + obj, "wb") as fp:
            pickle.dump(to_save, fp)

        return to_save



