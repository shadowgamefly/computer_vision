import os
import skimage
import pylab
import numpy as np
import math, random
import cv2
from skimage import data
from skimage.color import rgb2grey
from skimage import filters, io
from scipy.signal import convolve2d
from scipy import ndimage as ndi
import skimage.transform

def apply(H, a):
    np.set_printoptions(suppress=True)
    a.append(1)
    a = np.dot(H, a)
    a[:] = [x / a[-1] for x in a]
    return [a[0], a[1]]

def fits(A_list, B_list):
    np.set_printoptions(suppress=True)
    A = np.zeros([8, 8])
    b = np.zeros([8])
    for i in range(4):
        pA = A_list[i]
        pB = B_list[i]
        A[2*i] = [pB[0], pB[1], 1, 0 , 0, 0, -1 * pA[0] * pB[0], -1*pA[0]*pB[1]]
        b[2*i] = pA[0]
        A[2*i + 1] = [0, 0, 0, pB[0], pB[1], 1, -1 * pA[1] * pB[0], -1 * pA[1] * pB[1]]
        b[2*i + 1] =pA[1]
    x = np.linalg.solve(A, b)
    epi = 1e-10
    for i in x:
        if i < epi:
            i = 0
    return np.append(x, 1).reshape([3,3])

def composite_warped(a, b, H):
    "Warp images a and b to a's coordinate system using the homography H which maps b coordinates to a coordinates."
    out_shape = (a.shape[0], 2*a.shape[1])                               # Output image (height, width)
    p = skimage.transform.ProjectiveTransform(np.linalg.inv(H))       # Inverse of homography (used for inverse warping)
    bwarp = skimage.transform.warp(b, p, output_shape=out_shape)         # Inverse warp b to a coords
    bvalid = np.zeros(b.shape, 'uint8')                               # Establish a region of interior pixels in b
    avalid = np.zeros(a.shape, 'uint8')
    bvalid[1:-1,1:-1,:] = 255
    avalid[1:-1,1:-1,:] = 255
    bmask = skimage.transform.warp(bvalid, p, output_shape=out_shape)    # Inverse warp interior pixel region to a coords
    overlap=np.zeros((a.shape[0], 2*a.shape[1]))
    for i in range(402):
        for j in range(602):
            if bmask[i][j][0] != 0:
                overlap[i][j] = 1
    apad = np.hstack((skimage.img_as_float(a), np.zeros(a.shape))) # Pad a with black pixels on the right
    origin_image = (np.where(bmask==1.0, bwarp, apad))    # Select either bwarp or apad based on mask
    onlyA = np.ones((a.shape[0], 2*a.shape[1]))
    for i in range(402):
        for j in range(602):
            if overlap[i][j] == 0:
                onlyA[i][j] = 0;

    onlyB = np.ones((a.shape[0], 2*a.shape[1]))
    for i in range(402):
        for j in range(602, 1204):
            if bmask[i][j][0] != 0:
                onlyB[i][j] = 0;

    DistanceToA = ndi.morphology.distance_transform_edt(onlyA)
    DistanceToB = ndi.morphology.distance_transform_edt(onlyB)
    newA = skimage.img_as_float(a)
    for i in range(402):
        for j in range(604):
            if overlap[i][j] == 1:
                ratio = DistanceToA[i][j] / (DistanceToB[i][j]+DistanceToA[i][j])
                origin_image[i][j] = (1-ratio) * newA[i][j] + ratio * origin_image[i][j]

    return skimage.img_as_ubyte(origin_image)

image_a = cv2.imread("a.jpg")
image_b = cv2.imread("b.jpg")

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(image_a,None)
kp2, des2 = sift.detectAndCompute(image_b,None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

ITE = 500
T = 0.005
H_list = np.zeros([ITE, 3, 3])
max_inlier = 0
max_index = -1
for i in range(ITE):
    pairs = random.sample(good, 4)
    A_list = list([kp1[match[0].queryIdx].pt[0], kp1[match[0].queryIdx].pt[1]] for match in pairs)
    B_list = list([kp2[match[0].trainIdx].pt[0], kp2[match[0].trainIdx].pt[1]] for match in pairs)
    try:
        H_list[i] = fits(A_list, B_list)
    except:
        continue
    count = 0
    for match in good:
        a = [kp1[match[0].queryIdx].pt[0], kp1[match[0].queryIdx].pt[1]]
        b = [kp2[match[0].trainIdx].pt[0], kp2[match[0].trainIdx].pt[1]]
        if np.linalg.norm(np.subtract(apply(H_list[i], b), a)) < T:
            count += 1
    if count > max_inlier:
        max_index = i
        max_inlier = count

H = H_list[max_index]

final = composite_warped(image_a, image_b, H)

cv2.imshow("panaroma.png", final)
cv2.waitKey(0)
