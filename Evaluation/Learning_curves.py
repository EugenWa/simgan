# basic matrix operations, dataloading
import os
import sys

from glob import glob
import pickle

# - plotting, img tools -
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np
import random
import ntpath

import sys
sys.path.insert(0, '../')
#from Evaluation.Train_eval_1D_update import VAE_evaluation

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

colors_id       = ['mediumvioletred',   'purple',       'crimson',  'mediumblue',   'dodgerblue',   'cyan']
colors_no       = ['orange',            'darkorange',   'gold',     'lawngreen',    'springgreen',  'sandybrown']
img_line_styles     = ['-', ':', '-.', '--']
img_markers         = ['o', 'x']

SAVE_PATH = None

def load_validation_data_VAE(path_to_validations):
    # read all the paths
    paths = glob(path_to_validations + '/*')
    paths = [p for p in paths if not path_leaf(p).__contains__('BEST_')]

    # load Mean losses
    validation_history = []
    for path in paths:
        P_tmp = path_leaf(path)
        P_tmp = P_tmp.replace('NO', '')     # removes the thing in the pathname
        P_tmp = P_tmp.replace('ID', '')
        epoch = int(P_tmp)
        with open(path + '/Mean_loss', "rb") as fp:
            validation_history.append([epoch, pickle.load(fp)])

    validation_history = sorted(validation_history, key=lambda x:x[0])
    return validation_history

def validation_data_VAE(path_to_validations):
    paths = glob(path_to_validations + '/*')
    validation_history_id = load_validation_data_VAE([p for p in paths if path_leaf(p).__contains__('ID')][0])
    validation_history_no = load_validation_data_VAE([p for p in paths if path_leaf(p).__contains__('NO')][0])
    return validation_history_id, validation_history_no

def read_trainin_history_file(history_path):
    with open(history_path, "rb") as fb:
        history = pickle.load(fb)
    return history

def Read_VAE_history(history_path):
    # read all the paths
    paths = glob(history_path + '/*')
    history_id = read_trainin_history_file([p for p in paths if path_leaf(p).__contains__('ID')][0])
    history_no = read_trainin_history_file([p for p in paths if path_leaf(p).__contains__('NO')][0])

    if history_path.__contains__('genlike'):
        ges_loss = []
    else:
        ges_loss = read_trainin_history_file(history_path + '/GES_LOSS')
    return history_id, history_no, ges_loss




##################################################################################################################

def Read_GAN_history(history_path, dataframe_mode=0):
    history_id_path  = history_path + '/ID_LOSS_GAN'
    history_ges_path = history_path + '/GES_LOSS_GAN'

    history_id  = read_trainin_history_file(history_id_path)
    history_ges = read_trainin_history_file(history_ges_path)


    # split the data into the necessary containers
    # -ID -
    loss_id_ges  = [l[0] for l in history_id['loss']]
    loss_id_gen  = [l[1] for l in history_id['loss']]
    loss_id_disc = [l[2] for l in history_id['loss']]

    #val_loss_ges        =  # gen*10 + disc*1
    val_loss_id_gen     = [l[0] for l in history_id['val_loss']]
    val_loss_id_disc    = [l[1] for l in history_id['val_loss']]
    val_loss_id_discACC = [l[2] for l in history_id['val_loss']]

    standart_deviation_id = history_id['std']


    # ges
    loss_ges_ges = [l[0] for l in history_ges['loss']]
    loss_ges_gen = [l[1] for l in history_ges['loss']]
    #loss_ges_fet = [l[2] for l in history_ges['loss']]
    loss_ges_disc = [l[2] for l in history_ges['loss']]

    val_loss_ges_gen = [l[0] for l in history_ges['val_loss']]
    val_loss_ges_disc = [l[1] for l in history_ges['val_loss']]
    val_loss_ges_discACC = [l[2] for l in history_ges['val_loss']]

    standart_deviation_ges = history_ges['std']

    print(history_id['loss'])
    print('-----------')
    print(loss_id_ges)
    print(loss_ges_ges)
    print(val_loss_id_gen)
    print('-----------')
    #print(len(loss_id_ges))
    #print(len(loss_ges_ges))



    generall_loss       = create_GAN_pd_data_frames_GESLOSS(loss_id_ges, loss_ges_ges, dataframe_mode)
    generator_loss      = create_GAN_pd_data_frames_GENLOSS(loss_id_gen, loss_ges_gen, val_loss_id_gen, val_loss_ges_gen, dataframe_mode)
    discriminator_loss  = create_GAN_pd_data_frames_DISCLOSS(loss_id_disc, loss_ges_disc, val_loss_id_disc, val_loss_ges_disc, dataframe_mode)
    discriminator_acc   = create_GAN_pd_data_frames_DISCACC(val_loss_id_discACC, val_loss_ges_discACC, dataframe_mode)

    return generall_loss, generator_loss, discriminator_loss, discriminator_acc

def create_GAN_pd_data_frames_GESLOSS(loss_id_ges, loss_ges_ges, mode=0):
    if mode is 0:
        pol1 = 0
        pol2 = 1
    else:
        pol1 = -1
        pol2 = 0
    df_ges_losses_id    = pd.DataFrame({'epoch':range(0, len(loss_id_ges)), 'GES-Loss ID':loss_id_ges})
    df_ges_losses_ges   = pd.DataFrame({'epoch': range(len(loss_id_ges) + pol1*len(loss_ges_ges), len(loss_id_ges) + pol2*len(loss_ges_ges)), 'GES-Loss NO': loss_ges_ges})

    return df_ges_losses_id, df_ges_losses_ges

def create_GAN_pd_data_frames_GENLOSS(loss_id_gen, loss_ges_gen, val_loss_id_gen, val_loss_ges_gen, mode=0):
    if mode is 0:
        pol1 = 0
        pol2 = 1
    else:
        pol1 = -1
        pol2 = 0
    df_gen_losses_id    = pd.DataFrame({'epoch':range(0, len(loss_id_gen)), 'GENERATOR-Loss ID':loss_id_gen})
    df_gen_losses_ges   = pd.DataFrame({'epoch': range(len(loss_id_gen) + pol1*len(loss_ges_gen), len(loss_id_gen) + pol2*len(loss_ges_gen)), 'GENERATOR-Loss NO': loss_ges_gen})

    df_gen_val_losses_id    = pd.DataFrame({'epoch': range(0, len(val_loss_id_gen)), 'GENERATOR-ValLoss ID': val_loss_id_gen})
    df_gen_val_losses_ges   = pd.DataFrame({'epoch': range(len(val_loss_id_gen) + pol1*len(val_loss_ges_gen), len(val_loss_id_gen) + pol2*len(val_loss_ges_gen)), 'GENERATOR-ValLoss NO': val_loss_ges_gen})

    return df_gen_losses_id, df_gen_losses_ges, df_gen_val_losses_id, df_gen_val_losses_ges

def create_GAN_pd_data_frames_DISCLOSS(loss_id_disc, loss_ges_disc, val_loss_id_disc, val_loss_ges_disc, mode=0):
    if mode is 0:
        pol1 = 0
        pol2 = 1
    else:
        pol1 = -1
        pol2 = 0
    df_disc_losses_id    = pd.DataFrame({'epoch':range(0, len(loss_id_disc)), 'DISCRIMINATOR-Loss ID':loss_id_disc})
    df_disc_losses_ges   = pd.DataFrame({'epoch': range(len(loss_id_disc) + pol1*len(loss_ges_disc), len(loss_id_disc) + pol2*len(loss_ges_disc)), 'DISCRIMINATOR-Loss NO': loss_ges_disc})

    df_disc_val_losses_id = pd.DataFrame({'epoch': range(0, len(val_loss_id_disc)), 'DISCRIMINATOR-ValLoss ID': val_loss_id_disc})
    df_disc_val_losses_ges = pd.DataFrame({'epoch': range(len(val_loss_id_disc) + pol1*len(val_loss_ges_disc), len(val_loss_id_disc) + pol2*len(val_loss_ges_disc)),'DISCRIMINATOR-ValLoss NO': val_loss_ges_disc})

    return df_disc_losses_id, df_disc_losses_ges, df_disc_val_losses_id, df_disc_val_losses_ges

def create_GAN_pd_data_frames_DISCACC(val_loss_id_discACC, val_loss_ges_discACC, mode=0):
    if mode is 0:
        pol1 = 0
        pol2 = 1
    else:
        pol1 = -1
        pol2 = 0
    df_disc_acc_id    = pd.DataFrame({'epoch':range(0, len(val_loss_id_discACC)), 'DISC-Acc ID':val_loss_id_discACC})
    df_disc_acc_ges   = pd.DataFrame({'epoch': range(len(val_loss_id_discACC) + pol1*len(val_loss_ges_discACC), len(val_loss_id_discACC) + pol2*len(val_loss_ges_discACC)), 'DISC-Acc NO': val_loss_ges_discACC})

    return df_disc_acc_id, df_disc_acc_ges


def Plot_pandas_data_normal(dataframe_1, dataframe_2, name_tags, color_pallet_1, color_pallet_2, title, fig_title, Loss_type='MAE-Loss'):
    fig = plt.figure()
    plt.suptitle(title)
    plt.plot(name_tags[0], name_tags[1],     data=dataframe_1, linestyle=color_pallet_1[0], marker=color_pallet_1[1], color=color_pallet_1[3], alpha=color_pallet_1[2])
    plt.plot(name_tags[0], name_tags[2],     data=dataframe_2, linestyle=color_pallet_2[0], marker=color_pallet_2[1], color=color_pallet_2[3], alpha=color_pallet_2[2])
    plt.xlabel(name_tags[0])
    plt.ylabel(Loss_type)
    plt.legend()

    fig.savefig(SAVE_PATH + '/' + fig_title)


def Plot_Train_and_validation(dataframe_1, dataframe_2, dataframe_1_val, dataframe_2_val, name_tags_1, name_tags_2,
                         color_pallet_1, color_pallet_2, fig_title, titles=['Loss history', 'Validation-Loss history'], Loss_type='MAE-Loss'):
    fig = plt.figure(figsize=(10, 5))
    ax1  = plt.subplot(121)
    ax1.set_title(titles[0])
    plt.plot(name_tags_1[0], name_tags_1[1], data=dataframe_1, linestyle=color_pallet_1[0], marker=color_pallet_1[1], color=color_pallet_1[3], alpha=color_pallet_1[2])
    plt.plot(name_tags_1[0], name_tags_1[2], data=dataframe_2, linestyle=color_pallet_2[0], marker=color_pallet_2[1], color=color_pallet_2[3], alpha=color_pallet_2[2])
    plt.xlabel(name_tags_1[0])
    plt.ylabel(Loss_type)
    plt.legend()
    plt.legend()

    ax2 = plt.subplot(122)
    ax2.set_title(titles[1])
    ax2.plot(name_tags_2[0], name_tags_2[1], data=dataframe_1_val, linestyle=color_pallet_1[0], marker=color_pallet_1[1],color=color_pallet_1[3], alpha=color_pallet_1[2])
    plt.plot(name_tags_2[0], name_tags_2[2], data=dataframe_2_val, linestyle=color_pallet_2[0], marker=color_pallet_2[1],color=color_pallet_2[3], alpha=color_pallet_2[2])
    plt.xlabel(name_tags_2[0])
    plt.ylabel(Loss_type)
    plt.legend()
    plt.legend()

    fig.savefig(SAVE_PATH + '/' + fig_title)


def Plot_Discloss(dataframe_1, dataframe_2, dataframe_1_val, dataframe_2_val, name_tags_1, name_tags_2,
                         color_pallet_1, color_pallet_2, fig_title, titles=['DISC-Loss history', 'DISC-Validation-Loss history'], Loss_type='MAE-Loss'):
    fig = plt.figure(figsize=(10, 5))
    ax1  = plt.subplot(121)
    ax1.set_title(titles[0])
    plt.plot(name_tags_1[0], name_tags_1[1], data=dataframe_1, linestyle=color_pallet_1[0], marker=color_pallet_1[1], color=color_pallet_1[3], alpha=color_pallet_1[2])
    plt.plot(name_tags_1[0], name_tags_1[2], data=dataframe_2, linestyle=color_pallet_2[0], marker=color_pallet_2[1], color=color_pallet_2[3], alpha=color_pallet_2[2])
    plt.xlabel(name_tags_1[0])
    plt.ylabel(Loss_type)
    plt.legend()
    plt.legend()

    ax2 = plt.subplot(122)
    ax2.set_title(titles[1])
    ax2.plot(name_tags_2[0], name_tags_2[1], data=dataframe_1_val, linestyle=color_pallet_1[0], marker=color_pallet_1[1],color=color_pallet_1[3], alpha=color_pallet_1[2])
    plt.plot(name_tags_2[0], name_tags_2[2], data=dataframe_2_val, linestyle=color_pallet_2[0], marker=color_pallet_2[1],color=color_pallet_2[3], alpha=color_pallet_2[2])
    plt.xlabel(name_tags_2[0])
    plt.ylabel(Loss_type)
    plt.legend()
    plt.legend()

    fig.savefig(SAVE_PATH  + '/' + fig_title)



##################################################################################################################












def compare_validation_losses(model_names, subplot_id, validation_losses):
    pass

def compare_losses(model_names, subplot_id, losses):
    pass

if __name__=='__main__':
    result_path = '../Evaluation Results'
    model_to_analyze = 'GAN'
    Model_Path = result_path + '/' + model_to_analyze
    Models_to_compare = glob(Model_Path + '/*')

    model_names = []
    model_training_history = []
    for model in Models_to_compare:
        model_names.append([path_leaf(model), model])
        folders = glob(model + '/*')
        history_folder = [v for v in folders if v.__contains__('History')][0]
        try:
            if model_to_analyze is 'VAE':
                model_training_history.append(Read_VAE_history(history_folder))
            elif model_to_analyze is 'GAN':
                model_training_history.append(Read_GAN_history(history_folder))
        except(Exception):
            model_names.pop()

    if model_to_analyze is 'GAN':
        for model_id in range(len(model_names)):
            #model_id = 2
            fig_title = model_names[model_id][0]
            SAVE_PATH = model_names[model_id][1]
            print(fig_title)
            dframes_ges = model_training_history[model_id][0]
            dframes_gen = model_training_history[model_id][1]
            dframes_discloss = model_training_history[model_id][2]
            dframes_discACCs = model_training_history[model_id][3]


            # axes:
            name_tags_ges_loss      = ['epoch', 'GES-Loss ID', 'GES-Loss NO']

            name_tags_gen_loss1     = ['epoch', 'GENERATOR-Loss ID', 'GENERATOR-Loss NO']
            name_tags_gen_loss2     = ['epoch', 'GENERATOR-ValLoss ID', 'GENERATOR-ValLoss NO']

            name_tags_disc_loss1    = ['epoch', 'DISCRIMINATOR-Loss ID', 'DISCRIMINATOR-Loss NO']
            name_tags_disc_loss2    = ['epoch', 'DISCRIMINATOR-ValLoss ID', 'DISCRIMINATOR-ValLoss NO']

            name_tags_disc_acc      = ['epoch', 'DISC-Acc ID', 'DISC-Acc NO']

            # coloring
            color_Set = 2
            collor_pallet_1 = [img_line_styles[0], img_markers[0], 0.8, colors_id[color_Set]]
            collor_pallet_2 = [img_line_styles[0], img_markers[0], 0.8, colors_no[color_Set]]


            # actual plotting
            Plot_pandas_data_normal(dframes_ges[0], dframes_ges[1], name_tags_ges_loss, collor_pallet_1, collor_pallet_2, fig_title + 'GES', 'Ges-Loss-Test')
            Plot_Train_and_validation(dframes_gen[0], dframes_gen[1], dframes_gen[2], dframes_gen[3], name_tags_gen_loss1, name_tags_gen_loss2, collor_pallet_1, collor_pallet_2, fig_title + 'TLOSSVAL')
            Plot_Discloss(dframes_discloss[0], dframes_discloss[1], dframes_discloss[2], dframes_discloss[3], name_tags_disc_loss1, name_tags_disc_loss2, collor_pallet_1, collor_pallet_2, fig_title + 'DISCLOSS')
            Plot_pandas_data_normal(dframes_discACCs[0], dframes_discACCs[1], name_tags_disc_acc, collor_pallet_1, collor_pallet_2, fig_title + 'DISC_ACC', 'Discriminator-Accuracy')

    plt.show()



    exit()



    #+ 'vae_saquare_set_VAE_VAE_evaluation1D/Validations' + '/' + 'vae_saquare_set_VAE_vaeID'
    Paths_Validation = [v + '/Validations' for v in Models_to_compare]
    Paths_TraininHistory = []
    for p in Models_to_compare:
        folders = glob(p + '/*')
        folder = [v for v in folders if v.__contains__('History')][0]
        Paths_TraininHistory.append(folder)
    #print(Paths_Validation)
    #print(Paths_TraininHistory)

    #validation_history_id, validation_history_no = validation_data_VAE(model_training_history[1])
    history_id, history_no, ges_loss             = model_training_history[1]

    history_id['loss'] = [v[0] for v in history_id['loss']]
    history_no['loss'] = [v[0] for v in history_no['loss']]

    #history_id['val_loss'] = [v[0] for v in history_id['val_loss']]
    #history_no['val_loss'] = [v[0] for v in history_no['val_loss']]
    #history_id['val_loss'] = [v[0] for v in history_id['val_loss']]
    print(history_id['val_loss'])
    #exit()






    '''
    fig = plt.figure()
    # create dataframe
    df_id = pd.DataFrame({'epoch':[v[0] for v in validation_history_id], 'Validation-loss ID-mapping':[v[1] for v in validation_history_id]})
    df_no = pd.DataFrame({'epoch': [v[0] for v in validation_history_no], 'Validation-loss Domain-mapping': [v[1] for v in validation_history_no]})
    plt.plot('epoch', 'Validation-loss ID-mapping', data=df_id, marker='o', color='mediumvioletred', alpha=0.8)
    plt.plot('epoch', 'Validation-loss Domain-mapping', data=df_no, marker='o', color='orange', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MAE-Loss')
    plt.legend()
    '''
    # - training history
    fig_training_history = plt.figure(figsize=(10, 5))
    df_val_loss_id  = pd.DataFrame({'epoch':range(0, len(history_id['val_loss'])),
                                    'Validation-loss ID-mapping':history_id['val_loss']})
    df_val_loss_no  = pd.DataFrame({'epoch':range(len(history_id['val_loss']), len(history_id['val_loss']) + len(history_no['val_loss'])),
                                    'Validation-loss Domain-mapping':history_no['val_loss']})
    df_loss_id      = pd.DataFrame({'epoch': range(0, len(history_id['loss'])),
                                   'Loss ID-mapping': history_id['loss']})
    df_loss_no      = pd.DataFrame({'epoch': range(len(history_id['loss']), len(history_id['loss']) + len(history_no['loss'])),
                                    'Loss Domain-mapping': history_no['loss']})

    plt.subplot(121)
    plt.suptitle('Validation-Loss history')
    plt.plot('epoch', 'Validation-loss ID-mapping',     data=df_val_loss_id, marker='o', color='mediumvioletred', alpha=0.8)
    plt.plot('epoch', 'Validation-loss Domain-mapping', data=df_val_loss_no, marker='o', color='orange', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MAE-Loss')
    plt.legend()

    plt.subplot(122)
    plt.suptitle('Loss history')
    plt.plot('epoch', 'Loss ID-mapping',     data=df_loss_id, marker='o', color='mediumvioletred', alpha=0.8)
    plt.plot('epoch', 'Loss Domain-mapping', data=df_loss_no, marker='o', color='orange', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MAE-Loss')
    plt.legend()

    plt.show()