import os
import skimage
from scipy import ndimage as ndi
import pylab
from matplotlib import pyplot as plt
from numpy import linalg as LA
import numpy as np
import math
from skimage import data
from skimage.color import rgb2grey
from skimage import filters
from skimage import io
from scipy.signal import convolve2d

def filter_x(size, sigma):
    f = np.zeros([size * 2 + 1,size * 2 + 1])
    for i in range(-size, size + 1) :
        for j in range(-size, size + 1) :
            f[i + size][j + size] = (-i / (2 * math.pi * (sigma ** 4))) * math.exp(-(i*i + j*j)/2 / (sigma**2)) * 100
    return f

def filter_y(size, sigma):
    f = np.zeros([size * 2 + 1,size * 2 + 1])
    for i in range(-size, size + 1) :
        for j in range(-size, size + 1) :
            f[i + size][j + size] = (-j / (2 * math.pi * (sigma ** 4))) * math.exp(-(i*i + j*j)/2 / (sigma**2)) * 100
    return f

filename = input("The file you want to test: ")
T = float(input("The corner threshold: "))
image = io.imread(filename)
image=skimage.img_as_float(image)
image_grey = rgb2grey(image)

gaussian_x = filter_x(2, 3)
gaussian_y = filter_y(2, 3)
image_x = convolve2d(image_grey, gaussian_x)
image_y = convolve2d(image_grey, gaussian_y)
image_C = np.zeros(image_grey.shape)
image_removed = np.ones(image_grey.shape)
corners = np.zeros(image_grey.shape)

dtype = [('x', int), ('y', int), ('value', float)]
ini = [(0, 0, 0)]
li = np.array(ini, dtype=dtype)

for i in range(4, image_grey.shape[0] - 4) :
    for j in range(4, image_grey.shape[1] - 4) :
        val1 = 0.0
        val2 = 0.0
        val3 = 0.0
        for p in range(-4, 5) :
            for q in range(-4, 5):
                val1 += image_x[i+p][j+q] * image_x[i+p][j+q]
                val2 += image_x[i+p][j+q] * image_y[i+p][j+q]
                val3 += image_y[i+p][j+q] * image_y[i+p][j+q]
        mat = np.array([[val1/81, val2/81] ,[val2/81, val3/81]])
        value, v = LA.eig(mat)
        result = np.sort(np.absolute(value))[0]
        if result > T:
            image_removed[i][j] = 0
            li = np.append(li, np.array([(i,j,result)], dtype=li.dtype))

coords = np.array([[0, 0]])
li = np.sort(li, order = 'value')[::-1]
size = 0
for i in range(0, li.shape[0]) :
    if image_removed[li[i][0]][li[i][1]] == 1 :
        continue
    elif li[i][0] < 20 and li[i][1] < 20 :
        continue
    else :
        coords = np.append(coords,[[li[i][0], li[i][1]]], axis=0)
        size += 1
        for p in range(-10, 10) :
            for q in range(-10, 10) :
                image_removed[li[i][0] + p][li[i][1] + q] = 1

coords = np.delete(coords, 0, 0)
fig, ax = plt.subplots()
ax.imshow(image)
ax.plot(coords[:, 1], coords[:, 0], '+r', markersize=10)
plt.show()
