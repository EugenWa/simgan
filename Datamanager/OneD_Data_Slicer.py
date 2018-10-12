import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from glob import glob
import os
import ntpath

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_all_img_paths(data_set_name):
    Datasetpath = os.path.dirname(os.path.abspath(__file__))
    Datasetpath = Datasetpath[0:-(len("Datamanager") + 1)]
    Datasetpath += '/Datasets'

    depth_paths     = glob(Datasetpath + '/%s/depth/*' % data_set_name)
    depth_gt_paths  = glob(Datasetpath + '/%s/depth_gt/*' % data_set_name)

    depth_paths     = sorted([p for p in depth_paths if not p.__contains__('_000')])
    depth_gt_paths  = sorted([p for p in depth_gt_paths if not p.__contains__('_000')])

    print('Dataset lengths: ')
    print('Depth:       ', len(depth_paths))
    print('Depth_Gt:    ', len(depth_gt_paths))

    return depth_paths, depth_gt_paths

def load_paths_into_data_matrix(depth_paths, depth_gt_paths, random_crop=True):
    basic_img = plt.imread(depth_paths[0])

    # calculate a max range that can be divided by 2 at least 4 times
    acceptable_img_width = basic_img.shape[0]
    acceptable_img_height = basic_img.shape[1]
    print('Default Image width:  ', acceptable_img_width)
    print('Default Image height: ', acceptable_img_height)
    min_h_w = min(acceptable_img_width, acceptable_img_height)
    acceptable_img_width = min_h_w
    acceptable_img_height = min_h_w
    number_of_divisions = 5
    # itterate
    while acceptable_img_width%(2**number_of_divisions) != 0:
        acceptable_img_width -= 1
    while acceptable_img_height%(2**number_of_divisions) != 0:
        acceptable_img_height -= 1

    print('Recalc Image width:  ', acceptable_img_width)
    print('Recalc Image height: ', acceptable_img_height)


    print('Min')
    print('Recalc Image width:  ', acceptable_img_width)
    print('Recalc Image height: ', acceptable_img_height)
    depth_img_tensor = np.zeros((len(depth_paths), acceptable_img_width, acceptable_img_height))
    depth_gt_img_tensor = np.zeros((len(depth_gt_paths), acceptable_img_width, acceptable_img_height))

    difference_width    = basic_img.shape[0] - acceptable_img_width
    difference_height   = basic_img.shape[1] - acceptable_img_height
    for idx in range(len(depth_paths)):
        cropping_offset_w = 0
        cropping_offset_h = 0
        if random_crop:
            cropping_offset_w = np.random.randint(0, difference_width)
            cropping_offset_h = np.random.randint(0, difference_height)

        depth_img_tensor[idx]     = plt.imread(depth_paths[idx])[cropping_offset_w:cropping_offset_w+acceptable_img_width, cropping_offset_h:cropping_offset_h+acceptable_img_height]
        depth_gt_img_tensor[idx]  = plt.imread(depth_gt_paths[idx])[cropping_offset_w:cropping_offset_w+acceptable_img_width, cropping_offset_h:cropping_offset_h+acceptable_img_height]

    return depth_img_tensor, depth_gt_img_tensor

def slice_images(image_tensor_depth, image_tensor_depth_gt, mode='h'):
    img_width  =  image_tensor_depth.shape[1]
    img_height = image_tensor_depth.shape[2]
    img_number = image_tensor_depth.shape[0]
    if mode=='h':
        img_slices_depth = np.zeros((img_number * img_height, img_width))
        img_slices_depth_gt = np.zeros((img_number * img_height, img_width))
        for img_idx in range(img_number):
            for row_idx in range(img_height):
                img_slices_depth[img_idx * img_height + row_idx] = image_tensor_depth[img_idx, :, row_idx]
                img_slices_depth_gt[img_idx * img_height + row_idx] = image_tensor_depth_gt[img_idx, :, row_idx]
    else:
        img_slices_depth    = np.zeros((img_number* img_width, img_height))
        img_slices_depth_gt = np.zeros((img_number * img_width, img_height))
        for img_idx in range(img_number):
            for col_idx in range(img_width):
                img_slices_depth[img_idx * img_width + col_idx] = image_tensor_depth[img_idx, col_idx, :]
                img_slices_depth_gt[img_idx * img_width + col_idx] = image_tensor_depth_gt[img_idx, col_idx, :]

    img_slices_depth    = img_slices_depth[:, :, np.newaxis]
    img_slices_depth_gt = img_slices_depth_gt[:, :, np.newaxis]
    return img_slices_depth, img_slices_depth_gt

def complete_load(data_set_name, mode='h'):
    depth_paths, depth_gt_paths = get_all_img_paths(data_set_name)

    for i, j in zip(depth_paths, depth_gt_paths):
        # print(path_leaf(i), ' ----- ', path_leaf(j))
        if path_leaf(i) != path_leaf(j):
            print('Non, equal elements: ')
            print(path_leaf(i), ' ----- ', path_leaf(j))

    depth_img_tensor, depth_gt_img_tensor = load_paths_into_data_matrix(depth_paths, depth_gt_paths)

    return slice_images(depth_img_tensor, depth_gt_img_tensor, mode)


def disp_basic_img_data(img, disp=True):
    print(img.shape)
    print('Min: ', np.min(img))
    print('Max: ', np.max(img))
    print(img)

    if disp:
        fig = plt.figure()
        plt.imshow(img, cmap='gray')

if __name__=='__main__':
    data_set_name = 'brick_q'
    depth_slices_horizontal, depth_gt_slices_horizontal  = complete_load(data_set_name, 'hw')
    # save
    # -----------------------------------------------------------------------------------------
    Datasetpath = os.path.dirname(os.path.abspath(__file__))
    Datasetpath = Datasetpath[0:-(len("Datamanager") + 1)]
    Datasetpath += '/Datasets'

    path_A = Datasetpath + '/%s/A' % data_set_name
    path_B = Datasetpath + '/%s/B' % data_set_name
    os.makedirs(path_A, exist_ok=True)
    os.makedirs(path_B, exist_ok=True)
    config = np.array([depth_slices_horizontal.shape[0], depth_slices_horizontal.shape[1], depth_slices_horizontal.shape[1]])
    np.save(Datasetpath + '/%s/cfg' % data_set_name, config)

    print(depth_slices_horizontal.shape)

    np.save(path_A + '/A.npy', depth_gt_slices_horizontal)
    np.save(path_B + '/B.npy', depth_slices_horizontal)

    # visualize the depth slices
    # -----------------------------------------------------------------------------------------
    print('Number of all slices: ', depth_slices_horizontal.shape[0])
    select_rnd_image = np.random.randint(0, depth_slices_horizontal.shape[0])
    x_axis = range(0, depth_slices_horizontal.shape[1])
    plt.subplots(1, 3)
    plt.subplot(131)
    plt.plot(x_axis, depth_slices_horizontal[select_rnd_image], color='b')
    plt.title('DEPTH')

    plt.subplot(132)
    plt.plot(x_axis, depth_gt_slices_horizontal[select_rnd_image], color='r')
    plt.title('DEPTH_GT')

    plt.subplot(133)
    plt.plot(x_axis, depth_slices_horizontal[select_rnd_image], color='b')
    plt.plot(x_axis, depth_gt_slices_horizontal[select_rnd_image], color='r')

    print('')
    plt.show()