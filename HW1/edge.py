import os
import skimage
import pylab
import numpy as np
import math
from skimage import data
from skimage.color import rgb2grey
from skimage import filters
from skimage import io
from scipy.signal import convolve2d
from scipy import ndimage as ndi


# threshold for flower
# T_low = 0.12
# T_high = 0.2

# threshold for tower
# T_low = 0.5
# T_high = 1.5

def filter_x(size, sigma):
    f = np.zeros([size * 2 + 1,size * 2 + 1])
    for i in range(-size, size + 1) :
        for j in range(-size, size + 1) :
            f[i + size][j + size] = (-i / (2 * math.pi * (sigma ** 4))) * math.exp(-(i*i + j*j)/2 / (sigma**2))
    return f

def filter_y(size, sigma):
    f = np.zeros([size * 2 + 1,size * 2 + 1])
    for i in range(-size, size + 1) :
        for j in range(-size, size + 1) :
            f[i + size][j + size] = (-j / (2 * math.pi * (sigma ** 4))) * math.exp(-(i*i + j*j)/2 / (sigma**2))
    return f

def find_nearest(value):
    array = np.array([-math.pi, math.pi, -math.pi/2, -math.pi*3/4, -math.pi/4, 0, math.pi/4, math.pi/2, math.pi*3/4])
    idx = (np.abs(array-value)).argmin()
    if (idx == 0 or idx == 1) :
        return 0.0
    return array[idx]

filename = input("The file you want to test: ")
T_low = float(input("The low threshold: "))
T_high = float(input("The high threshold: "))

image = io.imread(filename)
image=skimage.img_as_float(image)
image_grey = rgb2grey(image)
T_high = 0.01
T_low = 0.005
gaussian_y = filter_x(3, 3)
gaussian_x = filter_y(3, 3)

image_x = convolve2d(image_grey, gaussian_x)
image_y = convolve2d(image_grey, gaussian_y)
image_F = np.zeros(image_grey.shape)
image_D = np.zeros(image_grey.shape)

for i in range(0, image.shape[0]):
    for j in range(0, image.shape[1]):
        image_F[i][j] = math.sqrt(image_x[i][j] * image_x[i][j] + image_y[i][j] * image_y[i][j])
        o = math.atan(image_y[i][j]/image_x[i][j])
        # print(find_nearest(orientation, o))
        image_D[i][j] = find_nearest(o)

image_I = np.empty(image_grey.shape)

for i in range(1, image.shape[0]-1):
    for j in range(1, image.shape[1]-1):
        currentO = image_D[i][j]
        currentF = image_F[i][j]
        if (currentO == 0 or currentO == math.pi) :
            if (currentF > image_F[i][j-1] and currentF > image_F[i][j+1]) :
                image_I[i][j] = currentF
            else :
                image_I[i][j] = 0;
        elif (currentO == math.pi/4 or currentO == -math.pi*3/4 ) :
            if (currentF > image_F[i+1][j+1] and currentF > image_F[i-1][j-1]) :
                image_I[i][j] = currentF
            else :
                image_I[i][j] = 0
        elif (currentO == math.pi/2 or currentO == -math.pi/2) :
            if (currentF > image_F[i+1][j] and currentF > image_F[i-1][j]) :
                image_I[i][j] = currentF
            else :
                image_I[i][j] = 0
        else :
            if (currentF > image_F[i-1][j+1] and currentF > image_F[i+1][j-1]) :
                image_I[i][j] = currentF
            else :
                image_I[i][j] = 0

edge = np.zeros(image_grey.shape)

for i in range(5, image.shape[0]-5):
    for j in range(5, image.shape[1]-5):
        if image_I[i][j] > T_high :
            edge[i][j] = 2
        elif image_I[i][j] < T_low :
            edge[i][j] = 0
        else :
            edge[i][j] = 1

updated = np.zeros(image_grey.shape)

for i in range(1, image_grey.shape[0] - 1) :
    for j in range(1, image_grey.shape[1] - 1) :
        if (edge[i][j] == 2 and updated[i][j] == 0):
            queue = np.array([[i,j]])
            updated[i][j] = 1
            link = 0
            while(queue.size != 0):
                link+=1
                current = queue[0]
                x = current[0]
                y = current[1]
                edge[x][y] = 2
                for p in range(-1, 2) :
                    for q in range(-1, 2) :
                         if (edge[x+p][y+q] > 0 and updated[x+p][y+q] == 0):
                             add = np.array([[x+p, y+q]])
                             queue = np.concatenate((queue, add))
                             updated[x+p][y+q] = 1
                queue = np.delete(queue, 0, axis = 0)

for i in range(0, image.shape[0]):
    for j in range(0, image.shape[1]):
        if edge[i][j] == 2 :
            edge[i][j] = 0
        else :
            edge[i][j] = 1

pylab.imshow(edge, cmap='Greys')
pylab.show()
