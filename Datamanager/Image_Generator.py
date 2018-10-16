import numpy as np
from scipy.misc import imread, imsave
import scipy
from matplotlib import pyplot as plt
from Data_sources.Noise_Models import *
from Data_sources.simple_math_operations import *
from Data_sources.Triangles import *
from Data_sources.Camera_Models import *
import os
from glob import glob
import math


'''
# example triangle
    v1 = np.array([2, 6, 6])
    v2 = np.array([6, 1, 5])
    v3 = np.array([2, 2, 2])
    '''
'''
v1 = np.array([0, 0, 1])
v2 = np.array([1.7, 1.7, 1])
v3 = np.array([-1, 2, 1])
'''

''' --- routines to generate random triangles --- '''
def create_rnd_tri_raycast(lock_distance=True, l_dist=3, allowed_overlap=0.91):
    cammax = 2
    camdispos = 1.5

    if lock_distance:
        distx3_v1 = l_dist
        distx3_v2 = l_dist
        distx3_v3 = l_dist
    else:
        distx3_v1 = np.random.uniform(2, 10)
        distx3_v2 = np.random.uniform(2, 10)
        distx3_v3 = np.random.uniform(2, 10)

    tmpv1 = (cammax * distx3_v1) / camdispos            # ohne camera disposition zu berücksichtigen
    tmpv2 = (cammax * distx3_v2) / camdispos
    v1 = np.random.uniform(tmpv1 * allowed_overlap, tmpv1 * allowed_overlap, (3,))
    v2 = np.random.uniform(tmpv2 * allowed_overlap, tmpv2 * allowed_overlap, (3,))
    v1[2] = distx3_v1
    v2[2] = distx3_v2
    while np.square(np.sum(v2 - v1)) < 1:
        v2 = np.random.uniform(tmpv2 * allowed_overlap, tmpv2 * allowed_overlap, (3,))
        v2[2] = distx3_v2

    tmpv3 = (cammax * distx3_v3) / camdispos
    v3 = np.random.uniform(tmpv3 * allowed_overlap, tmpv3 * allowed_overlap, (3,))
    v3[2] = distx3_v3
    while np.square(np.sum(v3 - v1)) < 5 or np.square(np.sum(v3 - v2)) < 5:
        v3 = np.random.uniform(tmpv3 * allowed_overlap, tmpv3 * allowed_overlap, (3,))
        v3[2] = distx3_v3

    return Triangle(v1, v2, v3)

def create_rnd_tri_indepth(lock_distance=True, l_dist=3, allowed_overlap=0.91):
    cammax = 2

    # set the distance coordinate for each corner
    if lock_distance:
        distx3_v1 = l_dist
        distx3_v2 = l_dist
        distx3_v3 = l_dist
    else:
        distx3_v1 = np.random.uniform(2, 10)
        distx3_v2 = np.random.uniform(2, 10)
        distx3_v3 = np.random.uniform(2, 10)

    v1 = np.random.uniform(-cammax*allowed_overlap, cammax*allowed_overlap, (3,))
    v2 = np.random.uniform(-cammax*allowed_overlap, cammax*allowed_overlap, (3,))
    v1[2] = 0
    v2[2] = 0
    v1[2] = distx3_v1
    v2[2] = distx3_v2
    while np.square(np.sum(v2 - v1)) < 0.5:
        v2 = np.random.uniform(-cammax*allowed_overlap, cammax*allowed_overlap, (3,))
        v2[2] = 0
        v2[2] = distx3_v2

    a = v2-v1

    v3 = np.random.uniform(-cammax*allowed_overlap, cammax*allowed_overlap, (3,))
    v3[2] = 0
    v3[2] = distx3_v3
    flag = True
    while np.square(np.sum(v3 - v1)) < 0.5 or np.square(np.sum(v3 - v2)) < 0.5 or flag:
        v3 = np.random.uniform(-cammax*allowed_overlap, cammax*allowed_overlap, (3,))
        v3[2] = 0
        v3[2] = distx3_v3
        #print("inloop")

        b = v3-v1
        #print(b)
        if (b[0] < 0.00000003) and (b[1] < 0.00000003):
            flag = True
        elif b[0] < 0.00000003:
            flag = False
        elif b[1] < 0.00000003:
            flag = False
        else:
            u1 = a[0]/b[0]
            u2 = a[1]/b[1]
            # recalc flag
            if u1-u2 < 0.28:
                flag = True
            else:
                flag = False

    return Triangle(v1, v2, v3)


''' - suplements the existing routines - '''
def create_raycast_angelrestricted(lock_distance, l_dist, allowed_overlap, min_angel=10):
    repetition_flag = True
    alignment_flag = True

    while repetition_flag or alignment_flag:
        tri = create_rnd_tri_raycast(lock_distance, l_dist, allowed_overlap)
        # check wether the angels are all reasonably high
        repetition_flag = check_angel_restriction(tri, min_angel)
        alignment_flag = check_flattnes_due_to_rotation_indepth(tri)

    return tri

def create_indepth_angelrestricted(lock_distance, l_dist, allowed_overlap, min_angel=10):
    repetition_flag = True
    alignment_flag = True

    while repetition_flag or alignment_flag:
        tri = create_rnd_tri_indepth(lock_distance, l_dist, allowed_overlap)
        # check wether the angels are all reasonably high
        repetition_flag = check_angel_restriction(tri, min_angel)
        alignment_flag = check_flattnes_due_to_rotation_indepth(tri)


    return tri

def check_angel_restriction(triangle, min_angel):
    a = np.sqrt(np.sum(np.square(triangle.geraden_tensor[0])))
    b = np.sqrt(np.sum(np.square(triangle.geraden_tensor[1])))
    c = np.sqrt(np.sum(np.square(triangle.geraden_tensor[2])))

    cos_gamma = (np.square(a) + np.square(b) - np.square(c)) / (2 * a * b)
    cos_alpha = (np.square(c) + np.square(b) - np.square(a)) / (2 * c * b)
    cos_beta  = (np.square(a) + np.square(c) - np.square(b)) / (2 * a * c)

    if math.degrees(abs(np.arccos(cos_gamma))) < min_angel:
        repetition_flag = True
    elif math.degrees(abs(np.arccos(cos_alpha))) < min_angel:
        repetition_flag = True
    elif math.degrees(abs(np.arccos(cos_beta))) < min_angel:
        repetition_flag = True                                  # all angels are at least greater than ten degrees
    else:
        repetition_flag = False

    return repetition_flag

def check_flattnes_due_to_rotation_indepth(triangle):
    print(np.dot(np.array([0, 0, 1]),triangle.norm))
    if abs(np.dot(np.array([0, 0, 1]), triangle.norm)) > 0.3:
        return False
    return True
'''
mode = 0 -> parallel raycasts
mode = 1 -> raycasts from cameracenter
'''
def create_random_triangle_set(amount, mode, lock_distance, lock_angels, perturb_l_dist=False):
    triangles = []
    if perturb_l_dist:
        l_dist = np.random.uniform(3, 9)
    else:
        l_dist = 3


    for i in range(amount):
        if mode is 1:
            if lock_angels:
                triangles.append(create_raycast_angelrestricted(lock_distance=lock_distance, l_dist=l_dist, allowed_overlap=1.13, min_angel=30))
            else:
                triangles.append(create_rnd_tri_raycast(lock_distance=lock_distance, l_dist=l_dist, allowed_overlap=1.13))
        elif mode is 0:
            if lock_angels:
                tri = create_indepth_angelrestricted(lock_distance=lock_distance, l_dist=l_dist, allowed_overlap=1.13, min_angel=30)
            else:
                tri = create_rnd_tri_indepth(lock_distance=lock_distance, l_dist=l_dist, allowed_overlap=1.13)
            triangles.append(tri)
    return triangles



''' - check for badly sampled triangles - '''
def check_image_integrity(img, trashhold=1, remove_too_slim_ones=True):
    img_max = np.max(img)
    trianglepartsH = []
    trianglepartsV = []
    for i in range(img.shape[0]):
        '''for j in range(img.shape[1]):
            if img[i, j] < img_max:
                triangleparts.append(i)
                break'''
        if np.mean(img[i, :]) < img_max:
            trianglepartsH.append(i)

    for j in range(img.shape[1]):
        '''for j in range(img.shape[1]):
            if img[i, j] < img_max:
                triangleparts.append(i)
                break'''
        if np.mean(img[:, j]) < img_max:
            trianglepartsV.append(j)

    if remove_too_slim_ones:
        if len(trianglepartsH) < 5 or len(trianglepartsV) < 5:
            return False

    # check for holes
    number_holes = 0
    for i in range(len(trianglepartsH) - 1):
        if trianglepartsH[i + 1] - trianglepartsH[i] > 1:
            number_holes += 1
    for i in range(len(trianglepartsV) - 1):
        if trianglepartsV[i + 1] - trianglepartsV[i] > 1:
            number_holes += 1

    if number_holes < trashhold:
        return True

    return False

''' - Triangle to Image translation - '''
def render_raycast_triangle_set(triangles, cam, check_sample_integrity=False):
    passing_triangles = []
    if check_sample_integrity:
        for tri in triangles:
            img = np.full((cam.imagedim[0], cam.imagedim[1]), 255)
            img = cam.raycast_render(tri, img)
            # check
            if check_image_integrity(img, 1):
                print('PASS')
                passing_triangles.append(tri)
            else:
                print('Triangle has holes or is too slim:')
    else:
        passing_triangles = triangles

    img = np.full((cam.imagedim[0], cam.imagedim[1]), 255)
    for tri in passing_triangles:
        img = cam.raycast_render(tri, img)
    return img

def render_indepth_triangle_set(triangles, cam, check_sample_integrity=False):
    # check if all triangles can be rendered in a reasonable manner
    passing_triangles = []
    if check_sample_integrity:
        for tri in triangles:
            img = np.full((cam.imagedim[0], cam.imagedim[1]), 255)
            img = cam.in_depth_render(tri, img)
            # check
            if check_image_integrity(img, 1):
                print('PASS')
                passing_triangles.append(tri)
            else:
                print('Triangle has holes or is too slim:')
    else:
        passing_triangles = triangles

    img = np.full((cam.imagedim[0], cam.imagedim[1]), 255)
    for tri in passing_triangles:
        img = cam.in_depth_render(tri, img)
    return img



''' - Data set generator - '''
def gendata_Normal(datasetname):
    cam = Camera()
    numofimgs = 12800
    for i in range(numofimgs):
        print("Tteration: ", i)
        try:
            tris = create_random_triangle_set(1)#(np.random.randint(1, 4))
            img = render_raycast_triangle_set(tris, cam)

            if np.min(img) < 240:
                imsave("images/%s/%d.png" % (datasetname, i), img)
            else:
                print("weird image in: ", i)
            #imsave(PATH_EE + "\\" + str(i) + ".png", img)
        except:
            print("minoreroor in: ", i)

def gendata_Noisy(datasetname):
    cam = Camera()
    noisem = Noise_mod()

    numofimgs = 12800
    for i in range(numofimgs):
        print("Tteration: ", i)
        try:
            tris = create_random_triangle_set(1)#(np.random.randint(1, 4))
            img = render_raycast_triangle_set(tris, cam)
            #add noise
            img = noisem.input_noise_edge(img)

            if np.min(img) < 240:
                imsave("images/%s/%d.png" % (datasetname, i), img)
            else:
                print("weird image in: ", i)
            #imsave(PATH_EE + "\\" + str(i) + ".png", img)
        except:
            print("minoreroor in: ", i)

def generate_paired_data_set(datasetname, data_type, lock_distance=True, lock_angels=False, Perturb_l_dist=False):
    # create directory
    os.makedirs('images/%s' % datasetname, exist_ok=True)
    domain_A_train = 'trainA'   # simulation
    domain_B_train = 'trainB'

    domain_A_test = 'testA'  # simulation
    domain_B_test = 'testB'

    os.makedirs('images/%s/%s' % (datasetname, domain_A_train), exist_ok=True)
    os.makedirs('images/%s/%s' % (datasetname, domain_B_train), exist_ok=True)
    os.makedirs('images/%s/%s' % (datasetname, domain_A_test), exist_ok=True)
    os.makedirs('images/%s/%s' % (datasetname, domain_B_test), exist_ok=True)

    # generate paired examples
    cam = Camera()
    noise = Noise_mod()

    number_of_train_images = 12800
    number_of_test_images = 1280*2

    triangle_amount = 1

    if data_type is 'parallel':
        # generate Trainingset
        for i in range(number_of_train_images):
            print("Trainset - Iteration: ", i)
            try:
                triangles = create_random_triangle_set(triangle_amount, 0, lock_distance, lock_angels, Perturb_l_dist)#(1, 0, lock_angels)
                img = render_indepth_triangle_set(triangles, cam, True)

                if np.min(img) < 210:
                    pic = scipy.misc.toimage(img, high=np.max(img), low=np.min(img))
                    # add_noise
                    img = noise.input_noise_edge(img, lock_distance)
                    img = noise.draw_lines(img, np.random.randint(1,4), 1, True, False, True)
                    img = noise.draw_ellipse_close_to_region_of_interest(img, 1)
                    picN = scipy.misc.toimage(img, high=np.max(img), low=np.min(img))
                    pic.save("images/%s/%s/%d.png" % (datasetname, domain_A_train, i))
                    picN.save("images/%s/%s/%d.png" % (datasetname, domain_B_train, i))
                else:
                    print("weird image in: ", i)
            except:
                print("minor error in: ", i)

        # generate testing set
        for i in range(number_of_test_images):
            print("Trainset - Iteration: ", i)
            try:
                triangles = create_random_triangle_set(triangle_amount, 0, lock_distance, lock_angels, Perturb_l_dist)#(1, 0, lock_angels)
                img = render_indepth_triangle_set(triangles, cam, True)

                if np.min(img) < 210:
                    pic = scipy.misc.toimage(img, high=np.max(img), low=np.min(img))

                    # add_noise
                    img = noise.input_noise_edge(img, lock_distance)
                    img = noise.draw_lines(img, np.random.randint(1, 4), 1, True, False, True)
                    img = noise.draw_ellipse_close_to_region_of_interest(img, 1)
                    picN = scipy.misc.toimage(img, high=np.max(img), low=np.min(img))
                    pic.save("images/%s/%s/%d.png" % (datasetname, domain_A_test, i))
                    picN.save("images/%s/%s/%d.png" % (datasetname, domain_B_test, i))
                else:
                    print("weird image in: ", i)
            except:
                print("minor error in: ", i)

def load_images_for_degrading(datasetname, DEGRADE_FURTHER=False):
    if DEGRADE_FURTHER:
        paths_train = glob('images/%s/%s/*' % (datasetname, 'trainB'))
        paths_test = glob('images/%s/%s/*' % (datasetname, 'testB'))
    else:
        paths_train = glob('images/%s/%s/*' % (datasetname, 'trainA'))
        paths_test = glob('images/%s/%s/*' % (datasetname, 'testA'))
    # load all data
    imgtrain = imread(paths_train[0])
    imgtest = imread(paths_test[0])
    train_images = np.zeros((len(paths_train), imgtrain.shape[0], imgtrain.shape[1]))
    test_images = np.zeros((len(paths_test), imgtest.shape[0], imgtest.shape[1]))

    # read images into matrix
    for i in range(len(paths_train)):
        train_images[i] = imread(paths_train[i])
    for i in range(len(paths_test)):
        test_images[i] = imread(paths_test[i])
    return train_images, test_images

def line_degradation(datasetname, fixed_Lineamount=True, DEGRADE_FURTHER=True):
    noise = Noise_mod()
    os.makedirs('images/%s/%s' % (datasetname, 'trainDFB'), exist_ok=True)
    os.makedirs('images/%s/%s' % (datasetname, 'testDFB'), exist_ok=True)

    if DEGRADE_FURTHER:
        paths_train = glob('images/%s/%s/*' % (datasetname, 'trainB'))
        paths_test = glob('images/%s/%s/*' % (datasetname, 'testB'))
    else:
        paths_train = glob('images/%s/%s/*' % (datasetname, 'trainA'))
        paths_test = glob('images/%s/%s/*' % (datasetname, 'testA'))
    # load all data
    imgtrain = imread(paths_train[0])
    imgtest = imread(paths_test[0])



    # determine usable paths
    usable_train_paths = []
    usable_test_paths = []
    for i in range(len(paths_train)):
        p = paths_train[i].split('/')
        p = [tmp for tmp in p if tmp.__contains__('.png')]
        p = p[0].split('\\')
        p = [tmp for tmp in p if tmp.__contains__('.png')][0]
        p = int(p.replace('.png', ''))
        usable_train_paths.append(p)
    for i in range(len(paths_test)):
        p = paths_test[i].split('/')
        p = [tmp for tmp in p if tmp.__contains__('.png')]
        p = p[0].split('\\')
        p = [tmp for tmp in p if tmp.__contains__('.png')][0]
        p = int(p.replace('.png', ''))
        usable_test_paths.append(p)

    train_images = np.zeros((max(usable_train_paths) + 1, imgtrain.shape[0], imgtrain.shape[1]))
    test_images = np.zeros((max(usable_test_paths) + 1, imgtest.shape[0], imgtest.shape[1]))

    # read images into matrix
    for i in range(len(paths_train)):
        train_images[usable_train_paths[i]] = imread(paths_train[i])

    for i in range(len(paths_test)):
        test_images[usable_test_paths[i]] = imread(paths_test[i])


    xAtrain = train_images
    xAtest = test_images
    test_and_train = True
    if test_and_train:
        for i in range(xAtrain.shape[0]):
            if i not in usable_train_paths:
                continue
            print('Adding lines to Train: ', i)
            if fixed_Lineamount:
                linemaount = 1
            else:
                linemaount = np.random.randint(1, 4)
            xAtrain[i] = noise.draw_lines(xAtrain[i], linemaount, 1, True, False, True)
            xAtrain[i] = noise.draw_ellipse_close_to_region_of_interest(xAtrain[i], 1)
            pic = scipy.misc.toimage(xAtrain[i], high=np.max(xAtrain[i]), low=np.min(xAtrain[i]))
            pic.save("images/%s/%s/%d.png" % (datasetname, 'trainDFB', i))
            # convert 2 picture
    for i in range(xAtest.shape[0]):
        if i not in usable_test_paths:
            continue
        print('Adding lines to Test: ', i)
        if fixed_Lineamount:
            linemaount = 1
        else:
            linemaount = np.random.randint(1, 4)
        xAtest[i] = noise.draw_lines(xAtest[i], linemaount, 1, True, False, True)
        xAtest[i] = noise.draw_ellipse_close_to_region_of_interest(xAtest[i], 1)
        pic = scipy.misc.toimage(xAtest[i], high=np.max(xAtest[i]), low=np.min(xAtest[i]))
        pic.save("images/%s/%s/%d.png" % (datasetname, 'testDFB', i))
    return xAtrain, xAtest

#Atrain, Atest = load_images_for_degrading('triangles_64_pertL')
#Atrain, Atest = line_degradation('triangles_64_pertL')
#fig = plt.figure()
#plt.imshow(Atrain[0], cmap='gray')
#plt.show()
#exit()


generate_paired_data_set('triangles_64_LinesEl', 'parallel', True, False, True)
'''
exit()
# gen dataset
DIR_PATH_NORMAL = "C:\\Users\\Tetra\\Desktop\\daten_sim"
DIR_PATH_28_NORMAL = "C:\\Users\\Tetra\\Desktop\\datasim28\\Normal"
DIR_PATH_28_NOISY = "C:\\Users\\Tetra\\Desktop\\datasim28\\Noisy"
DIR_PATH_280_NOISY = "C:\\Users\\Tetra\\Desktop\\datensim_noisy280"


# paths, necessary
p_norm_28mal2 = "trainA"
p_noisy_28mal2 = "trainB"
os.makedirs('images/%s' % p_norm_28mal2, exist_ok=True)
os.makedirs('images/%s' % p_noisy_28mal2, exist_ok=True)

gendata_Normal(p_norm_28mal2)
gendata_Noisy(p_noisy_28mal2)

'''
'''
cam = Camera()
tris = createrandom_triangles(1, 0)#np.random.randint(1, 4))
print("Aount:", len(tris))
img = create_image_from_triagles_indepth_render(tris, cam)
noisem = Noise_mod()
img = noisem.input_noise_edge(img)
fig = plt.figure()
print(img)
#tmp = img*(1/255)
#print(tmp)
# trafo
#img = (img-255)*(-1)
# scale
#img = img*(1/255)
#print(img)

plt.imshow(img, cmap='gray')
plt.show()
exit()
'''