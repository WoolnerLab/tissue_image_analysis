#!/usr/bin/env python
# encoding: utf-8
"""
graham_scan.py

Order the the convex hull of a polygon, given its vertices.

Created by Alexander Nestor-Bergmann on 2013-12-17.
Copyright (c) 2013 __MyCompanyName__. All rights reserved.
"""

import sys
import os
from math import atan2
from functools import reduce

LEFT_TURN = (1)

def cmp(a, b):
    return (a > b) - (a < b) 

def turn(a, b, c):
    # Compare the two possible directions: if '1' then turn is anticlockwise (defined above).
    return cmp((b[0] - a[0])*(c[1] - a[1]) - (c[0] - a[0])*(b[1] - a[1]), 0)


def keep_left(hull, c):
    while len(hull) > 1 and turn(hull[-2], hull[-1], c) != LEFT_TURN:
        hull.pop()
    if not len(hull) or hull[-1] != c:
        hull.append(c)
    return hull


def convex_hull(points):
    # Return the points given in an anticlockwise order around the convec hull.
    points = sorted(points)
    line = reduce(keep_left, points, [])
    line2 = reduce(keep_left, reversed(points), [])
    # Remember to leave out the first and last point when using extend on line
    return line.extend(line2[i] for i in range(1, len(line2) - 1)) or line


"""
Function to see if a point 'a' is clockwise or anticlockwise from a centroid, relative to a point 'b'.
"""


def less(a, b, centroid):

    # if a[0] - centroid[0] >= 0 and b[0] - centroid[0] < 0:
    # 	return True
    # if a[0] - centroid[0] < 0 and b[0] - centroid[0] >= 0:
    # 	return False
    #
    # if a[0] - centroid[0] == 0 and b[0] - centroid[0] == 0:
    # 	# if a[1] - centroid[1] >= 0 or b[1] - centroid[1] >= 0:
    # 	# 	return a[1] > b[1]
    # 	return b[1] > a[1]

    # Get the cross product of vectors: (centroid to a) X (centroid to b)
    crossProd = (a[0] - centroid[0]) * (b[1] - centroid[1]) - \
        (b[0] - centroid[0]) * (a[1] - centroid[1])
    if crossProd > 0:
        return False
    if crossProd < 0:
        return True

    # If it was == 0, then they are on the same line so check which was closer.
    aMag = (a[0] - centroid[0])**2 + (a[1] - centroid[1])**2
    bMag = (b[0] - centroid[0])**2 + (b[1] - centroid[1])**2
    return aMag < bMag


from numpy import asarray
from numpy import mean


def order_points(points):

    points = [list(t) for t in zip(*points)]
    points = asarray(points)

    centroid = [mean(points[0, :]), mean(points[1, :])]
    points = points.T

    # Get the angles each makes relative to the centroid.
    angles = [atan2(point[1] - centroid[1], point[0] - centroid[0]) for point in points]
    points = points.T

    # Sort in order of angles.
    indexlist = sorted(range(len(angles)), key=angles.__getitem__)
    # indexlist.reverse()
    points = points[:, indexlist]
    points = list(zip(points[0, :], points[1, :]))

    return points


def main():
    pass


if __name__ == '__main__':
    main()
