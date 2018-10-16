import scipy
from glob import glob
import numpy as np
import random
from scipy.misc import imsave, imread
from matplotlib import pyplot as plt
from Data_sources.Triangles import *
import ntpath


import PIL.ImageDraw as ImgDraw
import PIL.Image as Img
datasetname = 'triangles_64_LinesEl'
data_type = 'train'
paths_A = glob('./../datasets/%s/%sA/*' % (datasetname, data_type))
paths_B = glob('./../datasets/%s/%sB/*' % (datasetname, data_type))
#0.7058823 + 0.29411766
img = plt.imread(paths_A[0]) * (-1) + 1
print(img)
print()
print(img.shape, ' ', img.dtype)
print('Min: ', img.min(), ' Max: ', img.max())

img2 = plt.imread(paths_B[0]) * (-1) + 1

plt.subplot(1, 2, 1)
plt.imshow(img, cmap=plt.gray(), vmin=0, vmax=1)

ax  = plt.subplot(122)
plt.imshow(img2,  cmap=plt.gray(), vmin=0, vmax=1)
#plt.axis('off')
a = 12.1232354234234221
ax.set_ylabel('MAE Loss: ' + str(a)[0:7] + ' on image ' + str(12233))
ax.set_yticklabels([])

plt.suptitle('MAE Loss: ' + str(a)[0:7] + ' on image ' + str(123))
plt.show()
#
exit()
img = Img.new('P', (64, 64))
draw = ImgDraw.Draw(img)

img_center = (32, 32)
points = ((0 + 32, 0+ 32), (10 + 32, 0+ 32), (0+32, 22.5+32))
draw.polygon(points, fill=200)

plt.imshow(img)
#img.show()
plt.show()

#
exit()
from Data_sources.Image_Generator import *


cam = Camera()
noise = Noise_mod()

triangles = create_random_triangle_set(1, 0, True, True)#(1, 0, lock_angels)
img = render_indepth_triangle_set(triangles, cam)
while(np.min(img) > 210):
    triangles = create_random_triangle_set(1, 0, True, True)#(1, 0, lock_angels)
    img = render_indepth_triangle_set(triangles, cam, True)
    print('loop')

imgf = noise.mean_filter(img)
imgf = noise.draw_lines(imgf, 3, 1, normally_Distributed=False)
print(imgf.shape)
for i in range(imgf.shape[0]):
    print(i,  '    ', imgf[i, 0])
# clean imgf

fig = plt.figure()
plt.imshow(imgf, cmap='gray')
plt.show()
exit()

img2 = np.zeros(imgf.shape)
print(np.max(imgf))
print(np.min(imgf))
counter = 0
sum = 0
for i in range(imgf.shape[0]):
    for j in range(imgf.shape[1]):
        if imgf[i, j] > 0:
            print(counter, " - ", i, " ", j, ": ", imgf[i,j])
            print("-------", img[i, j])
            sum += imgf[i, j]
            counter += 1

        if abs(imgf[i, j]) > (avrg*1.28):

            img2[i, j] = 255
        else:
            imgf[i, j] = 0

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j] < 255:
            print(img[i, j])

img = noise.add_noise_std_with_trahshold(img, imgf, 38, avrg)
print('-----------------',np.max(img))
print(np.min(img))
print('SUM: ', sum)
print('AVRG:', sum/counter)
fig = plt.figure()
plt.imshow(img, cmap='gray')

fig = plt.figure()
plt.imshow(imgf, cmap='gray')

fig = plt.figure()
plt.imshow(img2, cmap='gray')
plt.show()


exit()

datasetname = 'triangles_64'
data_type = 'train'
paths_A = glob('../Data_sources/images/%s/%sA/*' % (datasetname, data_type))
paths_B = glob('../Data_sources/images/%s/%sB/*' % (datasetname, data_type))

# check integrety
items_A = set([])
items_B = set([])
for i in range(len(paths_A)):
    head, tail = ntpath.split(paths_A[i])
    items_A.add(tail)

for i in range(len(paths_B)):
    head, tail = ntpath.split(paths_B[i])
    items_B.add(tail)

# get difference
dif = items_A - items_B

print(len(dif))
#print(items_A.pop())
print('---------------------------------------')
print(paths_A[0])
head, tail = ntpath.split(paths_A[0])
print(tail, ntpath.basename(head))


img = imread(paths_A[0]).astype(np.uint8)
print(np.min(img))
scale = np.full((64, 64), 255)
#img = scale-img
print(img)
print(np.max(img))
for i in range(64):
    for j in range(64):
        if img[i,j] < 255:
            print(i, j, img[i, j])


fig = plt.figure()
plt.imshow(img, cmap='gray')

print('NEW IMG')
img = imread(paths_B[0])
scale = np.full((64, 64), 255)
#img = scale-img
print(img)
for i in range(64):
    for j in range(64):
        if img[i, j] < 255:
            print(img[i, j])
fig = plt.figure()

plt.imshow(img, cmap='gray')
plt.show()