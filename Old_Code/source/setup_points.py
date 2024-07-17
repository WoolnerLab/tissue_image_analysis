# encoding: utf-8
"""
setup_points.py
Sets up arrays of points using various methods:
- tracking cell nuclei from a tiff image.
- square lattice
- hexagonal lattice
- random array with certain density
- can also add a stretch to the points

Created by Alexander Nestor-Bergmann on 2013-12-18.
"""

import sys
import os
# import ezatrous
# import topin
import tifffile
import random
import numpy as np
import math
import cv2
from skimage import io
import skimage.morphology as sk
from skimage.filters import gaussian
from mahotas.morph import hitmiss
import scipy.ndimage.filters as filters
import trackpy as tp


def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2


def fuse(points, d):
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n):
                if dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count += 1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append((point[0], point[1]))
    return ret


def endPoints(skel):
    endpoint1 = np.array([[0, 0, 0], [0, 1, 0], [2, 1, 2]])
    endpoint2 = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 1]])
    endpoint3 = np.array([[0, 0, 2], [0, 1, 1], [0, 0, 2]])
    endpoint4 = np.array([[0, 2, 1], [0, 1, 2], [0, 0, 0]])
    endpoint5 = np.array([[2, 1, 2], [0, 1, 0], [0, 0, 0]])
    endpoint6 = np.array([[1, 2, 0], [2, 1, 0], [0, 0, 0]])
    endpoint7 = np.array([[2, 0, 0], [1, 1, 0], [2, 0, 0]])
    endpoint8 = np.array([[0, 0, 0], [2, 1, 0], [1, 2, 0]])
    ep1 = hitmiss(skel, endpoint1)
    ep2 = hitmiss(skel, endpoint2)
    ep3 = hitmiss(skel, endpoint3)
    ep4 = hitmiss(skel, endpoint4)
    ep5 = hitmiss(skel, endpoint5)
    ep6 = hitmiss(skel, endpoint6)
    ep7 = hitmiss(skel, endpoint7)
    ep8 = hitmiss(skel, endpoint8)
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    return ep


def pruning(skeleton, size):

    for i in range(1, size):
        endpoints = endPoints(skeleton)
        endpoints = np.logical_not(endpoints)
        skeleton = np.logical_and(skeleton, endpoints)
    return skeleton


""" Function to create skeleton image """


def skeleton(img):

    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # ret,img = cv2.threshold(img,2,255,0)
    # img = cv2.adaptiveBilateralFilter(img,(9,9),150)
    # img = cv2.GaussianBlur(img,(5,5),0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while(not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel




def read_nuclei_image(filename, neighborhood_size=8):
    # Read the line.
    image = io.imread(filename)
    nx, ny = image.shape

    # # blur = cv2.bilateralFilter(img,9,75,75)
    # blur = cv2.adaptiveBilateralFilter(image,(9,9),150)
    # blur = cv2.GaussianBlur(blur,(5,5),0)
    # # Get skeleton
    # image = skeleton(blur)

    # Set all active pixels to have value 1.
    image[image < max(image[image > 0])] = 0
    image[image > 0] = 1

    # Get the x,y coords of the points on the line.
    # topin_image = topin.topin2d(image,4,0)
    # y,x = np.nonzero(topin_image)

    threshold = 0
    image1 = gaussian(image, neighborhood_size)

    data_max = filters.maximum_filter(image1, neighborhood_size)
    maxima = (image1 == data_max)
    data_min = filters.minimum_filter(image1, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    y, x = np.nonzero(maxima)
    # Zip the points into tuples.
    tempPoints = np.dstack([x, y])[0]

    # Fuse the points that are really close to each other.
    points = fuse(tempPoints, neighborhood_size)

    return maxima, points


def detect_nuclei(filename, holesFilename=None, nucleus_size=5):

    # Read the file.
    nucleiImage = io.imread(filename)
    if holesFilename != None:
        holes = io.imread(holesFilename)
        image = nucleiImage + holes
    else:
        image = nucleiImage

    # # Set all active pixels to have value 1.
    # image[image < max(image[image > 0])] = 0
    # image[image > 0] = 1

    # Locate the nuclei with trackpy
    f = tp.locate(image, 5)
    seeds = image.copy()
    # seeds[:] = 0
    # seeds[map(int, f.y), map(int, f.x)] = 1
    seeds[0][:] = 0
    yVals, xVals = f.y.values.astype(int), f.x.values.astype(int)
    seeds[yVals, xVals] = 1

    # Get the actual Nuclei we Need
    f = tp.locate(nucleiImage, 5)
    nuclei = nucleiImage.copy()
    # nuclei[:] = 0
    # nuclei[map(int, f.y), map(int, f.x)] = 1
    nuclei[0][:] = 0
    yVals, xVals = f.y.values.astype(int), f.x.values.astype(int)
    nuclei[yVals, xVals] = 1
    points = zip(f.x.values, f.y.values)
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(seeds)
    # plt.show()

    return seeds, points, nuclei


def read_edges_image(filename, smooth=True, sizeOfDots=1, smoothing=2):
    # Read the line.
    image = io.imread(filename)
    nx, ny = image.shape

    # # blur = cv2.bilateralFilter(img,9,75,75)
    # blur = cv2.adaptiveBilateralFilter(image,(9,9),150)
    # blur = cv2.GaussianBlur(blur,(5,5),0)
    # # Get skeleton
    # image = skeleton(blur)

    # Set all active pixels to have value 1.

    # image = gaussian_filter(image, 2)
    image[image < np.mean(image[image > 0])/4] = 0
    image[image > 0] = 1

    # Get the x,y coords of the points on the line.
    # topin_image = topin.topin2d(image,4,0)
    # y,x = np.nonzero(topin_image)

    # if sizeOfDots > 1:
    #     neighborhood_size = sizeOfDots
    #     threshold = 0
    #     image = gaussian_filter(image, neighborhood_size)
    #
    #     data_max = filters.maximum_filter(image, neighborhood_size)
    #     maxima = (image == data_max)
    #     data_min = filters.minimum_filter(image, neighborhood_size)
    #     diff = ((data_max - data_min) > threshold)
    #     maxima[diff == 0] = 0
    #
    #     image = maxima
    #
    # Create a disk with a certain radius to do the erosion and dilation.
    if smooth:
        image = sk.binary_dilation(image, footprint=[[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    diskRadius = smoothing
    selem = sk.disk(diskRadius)
    image = sk.closing(image, selem)
    # Perform the closing operation using the disk -- This is a dilation followed by erosion.
    # out = sk.opening(img, selem)
    # image = sk.closing(image, selem)
    # from skimage.morphology import watershed, dilation, disk
    # image = dilation(image, disk(3))
    # Perform the skeletonization.
    if smooth:
        image = sk.skeletonize(image)
    image = pruning(image, 3)
    if not smooth:
        image = sk.medial_axis(image, return_distance=False)
    # image = pruning(image, 3)

    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.gcf()
    # plt.savefig(mydir+'skeleton.png')

    #

    y, x = np.nonzero(image)
    # Zip the points into tuples.
    points = np.dstack([x, y])[0]

    return image, nx, ny, points


"""
Generate a square or honeycomb lattice
The inputs are square or honeycomb as latticeType, and the width (nx), and the
spacing (the distance between each point horizontally.)
"""


def lattice(latticeType, nx, spacing):

    ny = nx

    # Generate a linear array of points.
    x = np.linspace(0, nx, nx/spacing)

    # Do a square lattice.
    if latticeType == 'square':
        points = [[elem, el] for elem in x for el in x]

    # Honeycomb Lattice:
    if latticeType == 'honeycomb':
        points = []

        fac = nx/(2*(nx/spacing-1))  # How much to shift every other row by.
        counter = 1

        for elem in x:
            for el in x:
                if counter % 2 == 0:
                    points.append([elem, el+fac])

                else:
                    points.append([elem, el])
            counter += 1

    return nx, ny, points


"""
Set up a matern hardcore distribution of points with a certain density. Input is xboundary
(nx), y-boundary (ny), the density at which it should stop searching (maxDensity), and the
minimum boundary for which each point occupies (boundary).
"""

# Define a function to calculate the Euclidean distance between 2 points.


def euc_dist(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def matern(nx, ny, maxDensity, boundary):

    # Initialise the list
    points = [[np.random.uniform(0, nx), np.random.uniform(0, ny)]]

    check = 1  # Checks if the point is valid or too close to another.
    density = 0  # for stopping when the spread is too dense.
    breaker = 0

    while not breaker:

        # Set up random numbers for x and y.
        x = np.random.uniform(0, (nx*100))/100.
        y = np.random.uniform(0, ny*100)/100.
        temp = [x, y]

        # reset the check
        check = 1

        # Iterate through all of the points.
        for point in points:

            # Check if the Euclidean distance is within a limit
            if euc_dist(temp, point) < boundary:
                check = 0  # set so the point isn't added
                density += 1  # increase density counter.
                break

        if check:
            points.append(temp)
            density == 0

        # Set conditions for stopping the loop:
        if density == maxDensity:
            breaker = 1

    return points


"""
Induce a stretch in the points that are generated. Inputs are the data points, the factor
to stretch by in x direction (xStretch), and in the y direction (yStretch).
"""


def stretch(points, xStretch, yStretch, nx, ny):

    points = [[x[0] * xStretch, x[1] * yStretch] for x in points]

    nx *= xStretch
    ny *= yStretch

    return points


""" Main"""


def main():
    pass


if __name__ == '__main__':
    main()
