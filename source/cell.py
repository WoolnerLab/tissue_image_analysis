# -*- coding: utf-8 -*-
"""
Class to create a Cell based on its vertices.

Created on Sun Sep 29 17:40:26 2013

@author: Alex Nestor-Bergmann
"""

from source import inertia
from source import graham_scan
from source import mymath
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from scipy.spatial import Delaunay
import math

# Define the class.


class cell(object):

    """ Initialiser: """

    def __init__(self, inPoints, inJunctions, inInterior):

        self.vertices = inPoints
        self.triJunctions = inJunctions
        self.interior = inInterior
        self.neighbours = []
        self.isBoundaryCell = False
        # self.order_vertices()

        self.lineTension = -0.259
        self.contractility = 0.172

    """ Set up len() to return number of vertices """

    def __len__(self):

        return len(self.vertices)

    """ Get the edges """

    def get_edges(self, polygonise=True):

        self.edges = []

        count = 0  # A counter to stop when all edges are done.

        # First arrange the polygons so that the vertices are in order.
        if polygonise == False:
            self.vertices = graham_scan.order_points(self.vertices)
            # Now add the edge for the last and first vertex as this doesn't fit in the loop below.
            self.edges.append((np.array([self.vertices[-1][0], self.vertices[-1][1]]),
                               np.array([self.vertices[0][0], self.vertices[0][1]])))

            # Now loop over all vertices to create an edge
            while count < len(self.vertices)-1:
                self.edges.append((np.array([self.vertices[count][0], self.vertices[count][1]]),
                                   np.array([self.vertices[count+1][0], self.vertices[count+1][1]])))
                count += 1
        elif polygonise:
            jvs = graham_scan.order_points(list(self.triJunctions))
            # Now add the edge for the last and first vertex as this doesn't fit in the loop below.
            self.edges.append((np.array([jvs[-1][0], jvs[-1][1]]),
                               np.array([jvs[0][0], jvs[0][1]])))

            # Now loop over all vertices to create an edge
            while count < len(self.triJunctions)-1:
                self.edges.append((np.array([jvs[count][0], jvs[count][1]]),
                                   np.array([jvs[count+1][0], jvs[count+1][1]])))
                count += 1

        return self.edges

    """ Order the points """

    def order_vertices(self):

        self.vertices = graham_scan.order_points(self.vertices)

    """ Get the centroids of the Cells """

    def get_circularity(self, method='junctions', polygonise=True):

        if method == 'area':
            eVals, eVecs, angle = self.get_evals_evecs_area(polygonise)

        elif method == 'perim':
            eVals, eVecs, angle = self.get_evals_evecs_perim(polygonise)

        elif method == 'junctions':
            eVals, eVecs, angle = self.get_evals_evecs_junctions()

        self.circularity = eVals[1]/eVals[0]

        return self.circularity

    """ Get the centroids of the Cells """

    def get_centroid(self, method='junctions', polygonise=True):

        if polygonise:
            jvs = graham_scan.order_points(list(self.triJunctions))
        else:
            jvs = self.vertices

        self.centroid = inertia.get_centroid(jvs, method)

        return self.centroid

    """ Get the eigenvalues and vectors of each polygon """

    def get_evals_evecs(self):

        # Principal Directions:
        self.eVals, self.eVecs, self.angle = inertia.principal_axes(self.vertices)
        # self.eVals, self.eVecs, self.angle = inertia.principal_axes(np.array(list(self.triJunctions)))

        return self.eVals, self.eVecs, self.angle

    def get_evals_evecs_area(self, polygonise=True):

        # # Get the centroids of each polygon:
        # centroid = [np.mean(self.interior[:,0]), np.mean(self.interior[:,1])]
        #
        # # Define the shape tensor
        # shape = np.array([[0.,0.],[0.,0.]])
        # for point in self.interior:
        #
        #     # R^i
        #     r_ix = np.array([ point[0] - centroid[0]])
        #     r_iy = np.array([ point[1] - centroid[1]])
        #
        #     # Update shape tensor.
        #     shape[0,0] += r_ix*r_ix
        #     shape[0,1] += r_ix*r_iy
        #     shape[1,0] += r_iy*r_ix
        #     shape[1,1] += r_iy*r_iy
        #
        # matrix = (1./len(self.interior)) * shape
        #
        # # Get the Eigenvalues/vectors
        # eVals, eVecs = np.linalg.eig(matrix)
        # indexList = np.argsort(-eVals)
        # eVals = eVals[indexList]
        # eVecs = eVecs[:,indexList]
        #
        #
        # angle = np.arctan2(eVecs[1][0], eVecs[0][0])
        # if angle < 0: angle += np.pi
        # angle *= 180/np.pi
        #
        # self.eVals, self.eVecs, self.angle = eVals, eVecs, angle

        # Get the centroids of each polygon:
        # centroid = [np.mean(self.interior[:,0]), np.mean(self.interior[:,1])]
        if polygonise == True:
            jvs = graham_scan.order_points(list(self.triJunctions))
            centroid = self.get_centroid(method='area', polygonise=True)
        elif polygonise == False:
            jvs = graham_scan.order_points(self.vertices)
            centroid = self.get_centroid(method='area', polygonise=False)

        # Define the shape tensor
        shape = np.array([[0., 0.], [0., 0.]])
        count = -1
        while count < len(jvs) - 1:

            # R^i
            r_ix = np.array([jvs[count][0] - centroid[0]])
            r_iy = np.array([jvs[count][1] - centroid[1]])

            # R^i+1
            r_ip1x = np.array([jvs[count+1][0] - centroid[0]])
            r_ip1y = np.array([jvs[count+1][1] - centroid[1]])

            # Area of subtriangle
            area = 0.5 * abs((r_ix*r_ip1y) - (r_iy*r_ip1x))

            # Update shape tensor.
            shape[0, 0] += (area/6)*(r_ip1x*r_ip1x + r_ix*r_ix) + \
                (area/12)*(r_ip1x*r_ix + r_ix*r_ip1x)
            shape[0, 1] += (area/6)*(r_ip1x*r_ip1y + r_ix*r_iy) + \
                (area/12)*(r_ip1x*r_iy + r_ix*r_ip1y)
            shape[1, 0] += (area/6)*(r_ip1y*r_ip1x + r_iy*r_ix) + \
                (area/12)*(r_ip1y*r_ix + r_iy*r_ip1x)
            shape[1, 1] += (area/6)*(r_ip1y*r_ip1y + r_iy*r_iy) + \
                (area/12)*(r_ip1y*r_iy + r_iy*r_ip1y)

            count += 1
        matrix = shape

        # Get the Eigenvalues/vectors
        eVals, eVecs = np.linalg.eig(matrix)
        indexList = np.argsort(-eVals)
        eVals = eVals[indexList]
        eVecs = eVecs[:, indexList]

        angle = np.arctan2(eVecs[1][0], eVecs[0][0])
        if angle < 0:
            angle += np.pi
        angle *= 180/np.pi

        self.eVals, self.eVecs, self.angle = eVals, eVecs, angle

        return self.eVals, self.eVecs, self.angle

    def get_evals_evecs_perim(self, polygonise=True):

        # Polygonsise or not.
        if polygonise == True:
            jvs = graham_scan.order_points(list(self.triJunctions))
            centroid = self.get_centroid(method='perim', polygonise=True)
        elif polygonise == False:
            jvs = graham_scan.order_points(self.vertices)
            centroid = self.get_centroid(method='perim', polygonise=False)

        # Define the shape tensor
        shape = np.array([[0., 0.], [0., 0.]])
        count = -1
        while count < len(jvs) - 1:

            # R^i
            r_ix = np.array([jvs[count][0] - centroid[0]])
            r_iy = np.array([jvs[count][1] - centroid[1]])

            # R^i+1
            r_ip1x = np.array([jvs[count+1][0] - centroid[0]])
            r_ip1y = np.array([jvs[count+1][1] - centroid[1]])

            # Length of edge.
            eLength = np.sqrt((jvs[count+1][0] - jvs[count][0])**2 +
                              (jvs[count+1][1] - jvs[count][1])**2)

            # Update shape tensor.
            shape[0, 0] += (eLength/3)*(r_ip1x*r_ip1x + r_ix*r_ix) + \
                (eLength/6)*(r_ip1x*r_ix + r_ix*r_ip1x)
            shape[0, 1] += (eLength/3)*(r_ip1x*r_ip1y + r_ix*r_iy) + \
                (eLength/6)*(r_ip1x*r_iy + r_ix*r_ip1y)
            shape[1, 0] += (eLength/3)*(r_ip1y*r_ip1x + r_iy*r_ix) + \
                (eLength/6)*(r_ip1y*r_ix + r_iy*r_ip1x)
            shape[1, 1] += (eLength/3)*(r_ip1y*r_ip1y + r_iy*r_iy) + \
                (eLength/6)*(r_ip1y*r_iy + r_iy*r_ip1y)

            count += 1
        matrix = shape

        # Get the Eigenvalues/vectors
        eVals, eVecs = np.linalg.eig(matrix)
        indexList = np.argsort(-eVals)
        eVals = eVals[indexList]
        eVecs = eVecs[:, indexList]

        angle = np.arctan2(eVecs[1][0], eVecs[0][0])
        if angle < 0:
            angle += np.pi
        angle *= 180/np.pi

        # eVecs = (eVecs * eVals)

        self.eVals, self.eVecs, self.angle = eVals, eVecs, angle

        return self.eVals, self.eVecs, self.angle

    def get_evals_evecs_junctions(self):

        # Get the centroids of each polygon:
        centroid = self.get_centroid(method='junctions')
        # centroid =  inertia.get_centroid(self.triJunctions)

        # Define the shape tensor
        shape = np.array([[0., 0.], [0., 0.]])
        jvs = graham_scan.order_points(list(self.triJunctions))
        for point in jvs:

            # length = np.sqrt((point[0] - centroid[0])**2 + point[1] - centroid[1])**2)
            # R^i
            r_ix = np.array([point[0] - centroid[0]])
            r_iy = np.array([point[1] - centroid[1]])

            # Update shape tensor.
            shape[0, 0] += r_ix*r_ix
            shape[0, 1] += r_ix*r_iy
            shape[1, 0] += r_iy*r_ix
            shape[1, 1] += r_iy*r_iy

        matrix = (1./len(self.triJunctions)) * shape

        # Get the Eigenvalues/vectors
        eVals, eVecs = np.linalg.eig(matrix)
        indexList = np.argsort(-eVals)
        eVals = eVals[indexList]
        eVecs = eVecs[:, indexList]

        angle = np.arctan2(eVecs[1][0], eVecs[0][0])
        if angle < 0:
            angle += np.pi
        angle *= 180/np.pi

        self.eVals, self.eVecs, self.angle = eVals, eVecs, angle

        return self.eVals, self.eVecs, self.angle

    """ Get the angles of the principal axes relative to the x axis """

    def get_axis_angle(self):

        Mect, Mcmp, self.axesAngles = inertia.second_order_moments(self.vertices)

        return self.axesAngles

    def get_pEff(self, a0=3355):
        ''' Function to return the effective pressure of a cell. '''

        area = self.get_area(polygonise=True)/a0
        perim = self.get_perimeter(polygonise=True)/np.sqrt(a0)
        l, g = self.lineTension, self.contractility
        l0 = -l/(2*g)
        pEff = area - 1 + (g*perim*(perim - l0))/(2*area)
        return pEff

    def get_forces(self, a0=3355):

        # jvs = graham_scan.order_points(list(self.triJunctions))
        #
        # # Get the x and y vertex coords.
        # x = np.array(zip(*jvs)[0])
        # y = np.array(zip(*jvs)[1])
        #
        # # Get the lengths of the edges
        # eLengths = np.zeros( x.size )
        # count = -1
        # while count < x.size - 1:
        #     eLengths[count] = 1./np.sqrt( (x[count] - x[count+1])**2 + (y[count] - y[count+1])**2  )
        #     count += 1
        #
        # # Get the cell centroid
        # centroid = self.get_centroid()
        # # Get the distances relative to the centroid.
        # x_xbar = np.array([ i-centroid[0] for i in x ])
        # y_ybar = np.array([ i-centroid[1] for i in y ])
        # # Get the shifted vertices
        # d = deque(x_xbar)
        # d.rotate(-1)  # rotate to the right
        # x_xbar2 = np.array(d)
        # d = deque(y_ybar)
        # d.rotate(-1)  # rotate to the right
        # y_ybar2 = np.array(d)
        #
        #
        # # Do the different parts of the stress tensor.
        # D1 = np.array([ [sum(eLengths*x_xbar*x_xbar2), sum(eLengths*x_xbar*y_ybar2) ], \
        #                 [sum(eLengths*y_ybar*x_xbar2), sum(eLengths*y_ybar*y_ybar2) ] ])
        # D2 = np.array([ [sum(eLengths*x_xbar2*x_xbar), sum(eLengths*x_xbar2*y_ybar) ], \
        #                 [sum(eLengths*y_ybar2*x_xbar), sum(eLengths*y_ybar2*y_ybar) ] ])
        # S1 = np.array([ [sum(eLengths*x_xbar*x_xbar), sum(eLengths*x_xbar*y_ybar) ], \
        #                 [sum(eLengths*y_ybar*x_xbar), sum(eLengths*y_ybar*y_ybar) ] ])
        # S2 = np.array([ [sum(eLengths*x_xbar2*x_xbar2), sum(eLengths*x_xbar2*y_ybar2) ], \
        #                 [sum(eLengths*y_ybar2*x_xbar2), sum(eLengths*y_ybar2*y_ybar2) ] ])
        #
        # # Add together
        # D = D1+D2
        # S = S1+S2
        #
        # # Get the areas
        # area = self.get_area()
        #
        # # Anisotropic Stress:
        # J = (1./area) * (D - S)

        jvs = graham_scan.order_points(list(self.triJunctions))

        # Get the x and y vertex coords.
        x = np.array(zip(*jvs)[0]) / np.sqrt(a0)
        y = np.array(zip(*jvs)[1]) / np.sqrt(a0)

        area = self.get_area() / a0
        perim = self.get_perimeter()/np.sqrt(a0)

        tangents_x = np.zeros(len(jvs))
        tangents_y = np.zeros(len(jvs))
        eLengths = np.zeros(len(jvs))
        count = -1
        while count < len(jvs) - 1:

            eLengths[count] = np.sqrt((x[count] - x[count+1])**2 + (y[count] - y[count+1])**2)

            tangents_x[count] = (x[count+1] - x[count]) / eLengths[count]
            tangents_y[count] = (y[count+1] - y[count]) / eLengths[count]
            count += 1

        J = (1./area) * np.array([
            [sum(-eLengths*tangents_x*tangents_x), sum(-eLengths*tangents_x*tangents_y)],
            [sum(-eLengths*tangents_y*tangents_x), sum(-eLengths*tangents_y*tangents_y)]])

        pTerm = (area - 1) * np.eye(2)
        # Get the contracility term:
        # First define the preferred perimeter.
        l0 = -self.lineTension/(2*self.contractility)
        cTerm = self.contractility * (perim - l0) * J

        # Get eigenvectors
        eVals, eVecs = np.linalg.eig(pTerm - cTerm)
        # print eVals

        # Sort biggest first.
        indexList1 = np.argsort(-abs(eVals))
        eVals = eVals[indexList1]
        eVecs = eVecs[:, indexList1]

        # Get the angle
        angle = np.arctan2(eVecs[1][0], eVecs[0][0])
        if angle < 0:
            angle += np.pi

        angle = angle * 180 / np.pi

        return eVals, eVecs, angle

    """ Function to get area of cell """

    def get_area(self, polygonise=True):

        if polygonise == True:
            self.area = inertia.area(graham_scan.order_points(list(self.triJunctions)))
        else:
            self.area = inertia.area(self.vertices)

        return self.area

    def get_area_interior(self):

        self.area = len(self.interior)

        return self.area

    """ Function to get perimeter of a cell. """

    def get_perimeter(self, polygonise=True):

        if polygonise == True:
            self.perimeter = inertia.perimeter(graham_scan.order_points(list(self.triJunctions)))
        else:
            self.perimeter = inertia.perimeter(self.vertices)

        return self.perimeter

    """ Plot the centroid """

    def plot_centroid(self, method='area', polygonise=True):

        centroid = self.get_centroid(method, polygonise)

        plt.plot(centroid[0], centroid[1])

    """ Plot the Vertices of the Voronoi cell """

    def plot_vertices(self):

        plt.plot(zip(*self.vertices)[0], zip(*self.vertices)[1], 'go')

    """ Plot principal axes of the cell. """

    def plot_pAxes(self, method='vertices', polygonise=True):

        if method == 'vertices':
            eVals, eVecs, angle = self.get_evals_evecs()
            col = 'green'
            fac = 1
            lw = 2
        elif method == 'area':
            eVals, eVecs, angle = self.get_evals_evecs_area(polygonise)
            # eVals, eVecs, angle = self.get_forces()
            col = 'red'
            fac = 1
            lw = 4
        elif method == 'perim':
            eVals, eVecs, angle = self.get_evals_evecs_perim(polygonise)
            col = 'blue'
            fac = 0.85
            lw = 3.25
        elif method == 'junctions':
            eVals, eVecs, angle = self.get_evals_evecs_junctions()
            col = 'yellow'
            fac = 0.7
            lw = 2.5
        else:
            print ('ERROR NEED TO SPECIFY CORRECT SHAPE TENSOR')

        if polygonise:
            jvs = list(self.triJunctions)
        else:
            jvs = self.vertices

        inertia.plot_axes(jvs, eVecs, method=method, colour=col, factor=fac, lw=lw)

    def plot_minor_pAxis(self, method='vertices', polygonise=True):

        if method == 'vertices':
            eVals, eVecs, angle = self.get_evals_evecs()
            col = 'yellow'
        elif method == 'area':
            eVals, eVecs, angle = self.get_evals_evecs_area(polygonise)
            col = 'red'
        elif method == 'perim':
            eVals, eVecs, angle = self.get_evals_evecs_perim(polygonise)
            col = 'blue'
        elif method == 'junctions':
            eVals, eVecs, angle = self.get_evals_evecs_junctions()
            col = 'green'
        else:
            print ('ERROR NEED TO SPECIFY CORRECT SHAPE TENSOR')

        if polygonise:
            jvs = list(self.triJunctions)
        else:
            jvs = self.vertices

        inertia.plot_minor_axis(jvs, eVecs, method=method)
