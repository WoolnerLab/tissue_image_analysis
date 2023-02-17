# encoding: utf-8
"""
mymath.py

Mathematical functions for arrays and lists in python.

Created by Alexander Nestor-Bergmann on 2013-12-17.
Copyright (c) 2013 __MyCompanyName__. All rights reserved.
"""

import math
import numpy as np
import matplotlib.pyplot as plt


""" Function to find the 2d Euclidean distance """


def euc_dist_2d(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


""" Define a custom dot product as the build in version finds matrices unaligned."""


def dotProd_2d(a, b):
    return a[0]*b[0] + a[1]*b[1]


""" Function to get 2d magnitude squared: ||a||^2"""


def mag2_2d(a):
    return dotProd_2d(a, a)


""" Function for the triple product using vector identity:"""
# a x (b x c) = (a.c)*b - (a.b)*c


def tripProd_2d(a, b, c):
    return dotProd_2d(a, c)*b - dotProd_2d(a, b)*c


""" Function for the magnitude of the cross product squared using the identiy:"""
# ||a x b||^2 = (|a|^2)*(|b|^2) - (a.b)^2


def magCross2_2d(a, b):
    return mag2_2d(a)*mag2_2d(b) - dotProd_2d(a, b)**2


""" Function to compute the principal componets of a data set (in matrix form).
Note, the columns of the input matrix should be dimensions, the rows datapoints.
The function returns the principal components (in columns) and their strengths"""


def PCA(data):

    # Deviation Matrix:
    devMatrix = (data - np.mean(data, axis=0))
    # Covariance Matrix:
    covMatrix = np.cov(devMatrix.T)

    # Eigenvalues & vectors
    eigs, principalComponents = np.linalg.eig(covMatrix)

    # Sort the principal components in decreasing order of corresponding eigenvalue:
    indexList = np.argsort(-eigs)
    eigs = eigs[indexList]
    principalComponents = principalComponents[:, indexList]

    # HOW TO USE AND PLOT:

    # # Scale the data into smaller dimensions - below, 2 is chosen
    # dims = 2
    # #principalComponents = principalComponents[:,:dims]
    # rescaled = np.dot(principalComponents.T, data.T).T

    #
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(rescaled[:,0], rescaled[:,1], color = 'blue')
    # # ax1.scatter(rescaled[:,0], rescaled[:,2], color = 'red')
    # # ax1.scatter(rescaled[:,0], rescaled[:,3], color = 'green')
    # ax1.scatter(data[:,0], data[:,1],  color='red')
    #
    # plt.plot([0,principalComponents[0,0]*5],[0,principalComponents[1,1]*5])
    #
    # plt.show()

    return eigs, principalComponents


"""# A function that can remove an array from a list, as this is not built into python: """


def rmArray(inList, inArray):
    count = 0  # A counter to tell the function when to stop.
    listSize = len(inList)  # Establish the total length of the list.
    while count != listSize and not np.array_equal(inList[count], inArray):
        count += 1
    if count != listSize:
        inList.pop(count)  # This is where the removal is done.
    # Also make an error message.
    else:
        print(inArray)  # Print it to let the user know what we are talking about.
        print (count)
        print (len(inList))
        raise ValueError('Error! The specified array is not in the list.')


""" Function to return the ideal bin width for a histogram.
See http://176.32.89.45/~hideaki/res/histogram.html """


def bin_width(data):

    # Don't add to the plot.
    plt.hold(False)

    # Get the upper and lower values of the data.
    upper = max(data)
    lower = min(data)

    # Set the maximum and minimum number of bins: (integers)
    minBins = 2
    maxBins = int(len(data)/2)

    # Set up an array of a range of bin sizes.
    bins = range(minBins, maxBins)
    bins = np.array(bins)

    # Now get a list of the bin sizes.
    binSize = (upper-lower)/bins
    # Set up the cost.
    Cost = np.zeros(shape=(np.size(binSize), 1))

    # Computation of the cost function
    for i in range(np.size(bins)):
        edges = np.linspace(lower, upper, bins[i]+1)  # Bin edges

        # Now count the number of datapoints in each bin.
        Npoints = plt.hist(data, edges)
        Npoints = Npoints[0]

        # Get the mean:
        average = np.mean(Npoints)
        # Get the variance
        var = sum((Npoints-average)**2) / bins[i]
        # Calculate the cost
        Cost[i] = (2*average-var) / ((binSize[i])**2)

    # Now we can get the optimal bin size by minimising the cost:
    minCost = min(Cost)
    # Find the location of minimumcost.
    index = np.where(Cost == minCost)
    index = int(index[0])
    # Define the optimum bin size.
    width = binSize[index]

    return width


""" Main """


def main():
    pass


if __name__ == '__main__':
    main()
