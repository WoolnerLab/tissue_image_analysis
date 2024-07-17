# encoding: utf-8
"""
inertia.py

Functions for finding the areas, centroids and moments of inertia of polygons in a list.
See: http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon

Also gets principak axes, and comes with a function to add them to a plot.

Created by Alexander Nestor-Bergmann on 2013-12-17.
Copyright (c) 2013 __MyCompanyName__. All rights reserved.
"""

import sys
import os
from source import graham_scan
from source import mymath
import numpy as np
import matplotlib.pyplot as plt


""" Method to return the perimiters of a list of polygons. """


def perimeter(polygon):

    # Do a for loop over every polygon using the graham scan algorithm to sort them.
    polygon = graham_scan.order_points(polygon)

    count = 0  # A counter to keep track of which vertex we are on.

    # Define the perimeter as the last vertex calculation as this can't be done in the loop.
    tPerimeter = mymath.euc_dist_2d(polygon[-1], polygon[0])

    while count < len(polygon)-1:
        tPerimeter += mymath.euc_dist_2d(polygon[count], polygon[count + 1])
        count += 1

    perimeter = tPerimeter

    return perimeter


""" Method to return a (list of) polygon's 'signed area' """


def area(polygon):

    # sort vertices.
    polygon = graham_scan.order_points(polygon)

    area = 0
    for i in range(0, len(polygon)):
        j = i % len(polygon) - 1
        area += polygon[i][0]*polygon[j][1]
        area -= polygon[j][0]*polygon[i][1]

    return abs(area*0.5)

    # # Do a for loop over every polygon using the graham scan algorithm to sort them.
    #     polygon = graham_scan.order_points(polygon)
    #
    #     count = 0 # A counter to keep track of which vertex we are on.
    # # Define the area as the last vertex calculation as this can't be done in the loop.
    #     tempArea = polygon[-1][0] * polygon[0][1]  -  polygon[0][0] * polygon[-1][1]
    #
    #     while count < len(polygon)-1:
    #         tempArea += polygon[count][0] * polygon[count+1][1]  -  polygon[count+1][0] * polygon[count][1]
    #         count += 1
    #
    #     area = tempArea
    #
    # # The actual area is actually 1/2 of the result.
    #     return 0.5 * area


""" Method to calculate the locations of the centrods. Returns coords as([x1,x2,x3],[y1,y2,y3])"""


def get_centroid(polygon, method='junctions'):

    # Get the ordered vertices.
    jvs = graham_scan.order_points(polygon)
    # If we are using junctions, its just the mean
    if method == 'junctions':
        return np.array([np.mean(list(zip(*jvs))[0]), np.mean(list(zip(*jvs))[1])])

    # Loop and get the centroid
    cPerim = np.array([0., 0.])
    cArea = np.array([0., 0.])
    denom = 0.
    count = -1
    while count < len(jvs) - 1:

        # R^i
        r_ix = jvs[count][0]
        r_iy = jvs[count][1]

        # R^i+1
        r_ip1x = jvs[count+1][0]
        r_ip1y = jvs[count+1][1]

        # Length of edge.
        eLength = np.sqrt((jvs[count+1][0] - jvs[count][0])**2 +
                          (jvs[count+1][1] - jvs[count][1])**2)

        # Update perim shape tensor.
        cPerim[0] += 0.5*eLength*(r_ix + r_ip1x)
        cPerim[1] += 0.5*eLength*(r_iy + r_ip1y)

        # # Update area shape tensor.
        # cArea[0] += eLength*((1./6.)*r_ix**2 + (1./6.)*r_ip1x*r_ix + (1./6.)*r_ip1x**2)
        # cArea[1] += eLength*((1./6.)*r_iy**2 + (1./6.)*r_ip1y*r_iy + (1./6.)*r_ip1y**2)
        # denom[0] += 0.25*eLength*(r_ix+ r_ip1x)
        # denom[1] += 0.25*eLength*(r_iy + r_ip1y)

        # Update area shape tensor.
        cArea[0] += (1./6.)*(r_ix + r_ip1x)*(r_ix*r_ip1y - r_ip1x*r_iy)
        cArea[1] += (1./6.)*(r_iy + r_ip1y)*(r_ix*r_ip1y - r_ip1x*r_iy)
        denom += 0.5*(r_ix*r_ip1y - r_ip1x*r_iy)

        count += 1

    if method == 'perim':
        return cPerim / perimeter(jvs)
    elif method == 'area':
        return cArea / denom

    # # xs, ys = zip(*polygon)
    # #
    # # return np.array([ np.mean(xs), np.mean(ys) ])
    #
    # # Do the graham scan algorithm to sort them.
    # polygon = graham_scan.order_points(polygon)
    #
    # count = 0 # Reset the counter every time.
    # # Set the x and y centroid ponts (as the last vertex calculation):
    # Cx = (polygon[-1][0] + polygon[0][0]) * (polygon[-1][0] * polygon[0][1] - polygon[0][0] * polygon[-1][1])
    # Cy = (polygon[-1][1] + polygon[0][1]) * (polygon[-1][0] * polygon[0][1] - polygon[0][0] * polygon[-1][1])
    #
    # while count < len(polygon)-1:
    # # Do the calculation for x and y.
    #     Cx += (polygon[count][0] + polygon[count + 1][0]) * (
    #             polygon[count][0] * polygon[count+1][1] - polygon[count+1][0] * polygon[count][1])
    #
    #     Cy += (polygon[count][1] + polygon[count + 1][1]) * (
    #             polygon[count][0] * polygon[count+1][1] - polygon[count+1][0] * polygon[count][1])
    #
    #     count += 1
    #
    # # Calculate six times the area of the polygon
    # area6 = area(polygon) * 6
    #
    # # Multiply everything by the area prefactor.
    # Cx *= (1/area6)
    # Cy *= (1/area6)
    #
    # return np.array([Cx,Cy])


"""
Method to get 2D central moment, which can be used to evaluate the inertia tensor.
input arguments are: polygons (as vertices), the x,y coords of its centriod (centroid),
the order of the x moment (p) the order of the y moment (q), and the order of z (r). Note
in 2D we just use the order or z to get 0 in i31,32,23,13

See http://progmat.uw.hu/oktseg/kepelemzes/lec13_shape_4.pdf page 4 for details on methods.

"""


def central_moment(polygon, centroid, p, q, r):

    polygon = graham_scan.order_points(polygon)
    # Get the inverse areas:
    _area = 1/area(polygon)

    count = 0  # Set up a counter.

    moment = 0  # Reset the value for the moment

    # Loop over every vertex.
    while count < len(polygon):

        # Calculate the inertial moment per vertex.
        moment += (0**r) * ((polygon[count][0] - centroid[0])**p) * (
            (polygon[count][1] - centroid[1])**q)

        count += 1  # increase count.

    moment *= _area  # Multiply by 1/area

    return moment


""" Method to find the moment of inertia tensor
The input is a list of polygons. It will return a lists of lists with the various moments
of inertia i.e. i11 (== ixx). Order is 11,12,13,21,22,23,31,32,33
"""


def iTensor(polygon):

    # First get the centroids:
    centroid = get_centroid(polygon, method='area')

    # Set up the moments of ineria
    iTensor = [central_moment(polygon, centroid, 0, 2, 0)]  # Make z moment 0.
    iTensor.append(central_moment(polygon, centroid, 1, 1, 0))
    iTensor.append(central_moment(polygon, centroid, 1, 1, 0))
    iTensor.append(central_moment(polygon, centroid, 2, 0, 0))

    # # Set up the moments of ineria
    # iTensor = [central_moment(polygons, centroids, 0, 2, 0)] # Make z moment 0.
    # iTensor.append(central_moment(polygons, centroids, 1, 1, 0))
    # iTensor.append(central_moment(polygons, centroids, 1, 1, 1)) # z moment nonzero, other moments don't matter
    # iTensor.append(central_moment(polygons, centroids, 1, 1, 0))
    # iTensor.append(central_moment(polygons, centroids, 2, 0, 0))
    # iTensor.append(central_moment(polygons, centroids, 1, 1, 1))
    # iTensor.append(central_moment(polygons, centroids, 1, 1, 1))
    # iTensor.append(central_moment(polygons, centroids, 1, 1, 1))
    # iTensor.append(central_moment(polygons, centroids, 2, 2, 0))

    return iTensor


""" Method to take a list of polygons and return their scaled principal axes. The function will
also scale the length of the principal axes according to the eigenvalues. """


def principal_axes(polygon):

    # Get the centroids of each polygon:
    centroid = get_centroid(polygon, method='area')

    # Work out the inertia tensor for each polygon.
    itensor = iTensor(polygon)
    # # Transpose it so that each row represents a polygon.
    # itensor = np.asarray(itensor).T

    # Make a matrix where every entry is a 2x2 inertia tensor for a polygon.
    matrix = [[itensor[0], itensor[1]], [itensor[2], itensor[3]]]

    # Get the Eigenvalues/vectors
    eVals, eVecs = np.linalg.eig(matrix)
    # sort the principal directions in decreasing order of corresponding eigenvalue
    # Note this returns them in the form [(x1,x2), (y1,y2)] NOT [(x1,y1), (x2,y2)]###
    # Scale the eigenvectors to Poinsots ellipsoid:
    eVals = 1/np.sqrt(eVals)
    indexList = np.argsort(-eVals)
    eVals = eVals[indexList]
    eVecs = eVecs.T[:, indexList]

    angle = np.arctan2(eVecs[1][0], eVecs[0][0])
    if angle < 0:
        angle += np.pi
    angle *= 180/np.pi

    # count = 0
    # while count < len(eVals):
    #     if eVals[count] > 0:
    #         eVals[count] = 1/np.sqrt(eVals[count])
    #     if eVals[count] < 0:
    #         eVals[count] = -(1/np.sqrt(eVals[count]))
    #     count += 1
    # Multiply the eigenvectors by their eigenvalues to represent their strength.
    # eVecs = (eVecs * eVals)

    return eVals, eVecs, angle


""" Subfunction to add the principal axes of polygons to a plot. The function takes a list of
polygons, and their principal axes as input. """


def plot_minor_axis(polygon, eVecs, method='area'):

    # First get the centroids of each polygon:
    centroid = get_centroid(polygon, method)
    polygon = np.array(polygon)

    # Add the locations of the centroids (as a white cross) to the plot.
    #plt.plot(centroids[0],centroids[1], 'wx')
    factor = 0.5 * min(max(polygon.T[0]) - min(polygon.T[0]), max(polygon.T[1]) - min(polygon.T[1]))
    eVecs *= factor
    # Now add the axes to the plot.
    ecount = 0

    lines = plt.plot([centroid[0] - eVecs[0][1], centroid[0] + eVecs[0][1]],
                     [centroid[1] - eVecs[1][1], centroid[1] + eVecs[1][1]])

    plt.setp(lines, color='black', linewidth=2.0)


def plot_axes(polygon, eVecs, colour='black', method='area', factor=1, lw=2):

    # First get the centroids of each polygon:
    centroid = get_centroid(polygon, method)
    polygon = np.array(polygon)

    # Add the locations of the centroids (as a white cross) to the plot.
    #plt.plot(centroids[0],centroids[1], 'wx')
    factor = factor * 0.5 * \
        min(max(polygon.T[0]) - min(polygon.T[0]), max(polygon.T[1]) - min(polygon.T[1]))
    eVecs *= factor
    # Now add the axes to the plot.
    ecount = 0

    lines = plt.plot([centroid[0] - eVecs[0][0], centroid[0] + eVecs[0][0]],
                     [centroid[1] - eVecs[1][0], centroid[1] + eVecs[1][0]])

    plt.setp(lines, color=colour, linewidth=0, alpha=0.85)

    # lines = plt.plot([centroid[0] - eVecs[0][ecount+1], centroid[0] + eVecs[0][ecount+1]], \
    #         [centroid[1] - eVecs[1][ecount+1], centroid[1] + eVecs[1][ecount+1]])
    # plt.setp(lines, color='y', linewidth=2.0)

    # Principal Axes
    #     topArrow = plt.arrow(centroid[0], centroid[1], eVecs[0][ecount+1], eVecs[1][ecount+1], head_width=3,\
    # head_length=3, length_includes_head=True)
    #     bottomArrow = plt.arrow(centroid[0], centroid[1], - eVecs[0][ecount+1], -eVecs[1][ecount+1], head_width=3, \
    # head_length=3,  length_includes_head=True)
    #
    #     plt.setp(topArrow, color='black', linewidth=1.5, alpha = 0.8)
    #     plt.setp(bottomArrow, color='black', linewidth=1.5, alpha = 0.8)
    #
    # # Minor axes
    #     topArrow = plt.arrow(centroid[0], centroid[1], eVecs[0][ecount], eVecs[1][ecount], head_width=2,\
    # head_length=2, length_includes_head=True)
    #     bottomArrow = plt.arrow(centroid[0], centroid[1], - eVecs[0][ecount], -eVecs[1][ecount], head_width=2, \
    # head_length=2,  length_includes_head=True)
    #
    #     plt.setp(topArrow, color='black', linewidth=1, alpha = 0.6)
    #     plt.setp(bottomArrow, color='black', linewidth=1, alpha = 0.6)

    # ecount += 2

    # No return value:
    return


""" Method to return certain second order moments of a list of polygons. Will return
eccentricity moment (Mect), compactness moment (Mcmp), and the angles their major axes
makes with the x-axis (angles)"""


def second_order_moments(polygon):

    # Get the inertia tensor:
    itensor = iTensor(polygon)

    # Calculate the angle that each polygon's major axis makes with the x-axis.
    # Undefined for objects with more than 2 axes of symmetry. The longer the shape,
    # the more accurate this value is.
    angle = abs(0.5*(np.arctan2((2*itensor[1]), (itensor[3] - itensor[0]))))
    angle *= 180 / np.pi

    # Calculate the compactness moment. Shows compactness - Mcmp = 1 for disk.
    Mcmp = (1/(2 * np.pi)) * (area(polygon)/(itensor[0] + itensor[3]))

    # Use second moments to get a normalised figure of eccentricity (shows eleongation)
    # Less robust that Mcmp. Mect = 0 for disk and Mect = 1 for line.
    # Accurate for elongated shapes but not for circular because of numerical
    # instability when u02 - u20 and u11 small.
    # This is the normalised difference: I(max) - I(min).
    Mect = (np.sqrt((itensor[0] - itensor[3])**2 + 4 *
                    itensor[1] * itensor[1]))/(itensor[0] + itensor[3])

    return Mect, Mcmp, angle


""" Function to ge the shape factor (relative to a circle) for a list of polygons """


def shape_factor(polygon):

    # Get the areas:
    area = area(polygon)
    # Get the perimeters:
    perimeter = perimeter(polygon)
    # Get the shape factor:
    sf = (np.pi * 4 * area) / (perimeter * perimeter)

    return sf


""" Main """


def main():
    pass


if __name__ == '__main__':
    main()
