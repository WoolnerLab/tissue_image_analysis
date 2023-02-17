# -*- coding: utf-8 -*-
"""
Class to create a Voronoi Tessellation of a set of points found from a tif file. 

Created on Sun Sep 29 17:40:26 2013

@author: Alex Nestor-Bergmann
"""

from source import cell
from source import mymath
import numpy as np
from scipy.spatial import Delaunay
import math


## Set up a test array of points. 
#points = np.random.rand(75, 2)
## points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Function to perform a Voronoi Tesselation:
def voronoi(inPoints):
	
	points = inPoints

	# Get the maximums and minums of the points
	xMax, yMax =  np.asarray(points).max(axis=0)[0], np.asarray(inPoints).max(axis=0)[1]
	xMin, yMin =  np.asarray(points).min(axis=0)[0], np.asarray(inPoints).min(axis=0)[1]	

	# Perform the Delaunay triangulation
	triang = Delaunay(inPoints)

	# Build a matrix of matrices where each sub matrix is a triangle, made up of its vertices.
	triList = triang.points[triang.simplices]

	# Make a matrix containing the first vertex of every triangle (0 to n). Note transpose for
	# maths functions later.
	A = triList[:,0,:].T
	# Make a matrix containing the second vertex of every triangle.
	B = triList[:,1,:].T
	# Make a matrix containing the third vertex of every triangle.
	C = triList[:,2,:].T

	# Using the circumcircle equations, find the circumcentre:
	#
	# First we transpose the system to place C at the origin.
	a = A - C
	b = B - C

	# Extract the circumcentre (p0) for every triangle (each row of p0 represents a triangle)
	p0 = mymath.tripProd_2d(mymath.mag2_2d(a) * b - mymath.mag2_2d(b) * a, a, b) / (2*mymath.magCross2_2d(a, b)) + C
	#print(p0)

	# # Now determine the voronoi edges - These are just the edges connecting the circumcentres:
	# # establish the points within the triangles.
	# voron = p0[:,triang.neighbors]
	#
	# # Some of the edges go to infinity (from the outermost points), so just delete these.
	# voron[:,triang.neighbors == -1] = np.nan
	
	""" Initialise the polygons """
	
	cells = [] # This stores the edges of the polygons created for every point in points.

	for point in inPoints:

		if xMin < point[0] < xMax and yMin< point[1] < yMax:

			polygon = [] #  Temporarily stores the vertices of a polygon until it is verified.
			count = 0 # This is to count the indices.
			polyInBoundary = True # Flag for if polygon is in boundary.

			# For finding the other triangles connected to the point.
			for i in triList:
				for j in i:
					if (point == j).all() and (p0[0,count], p0[1,count]) not in polygon:
						# If edge within loop, add it to the polygon.
						polygon.append((p0[0,count], p0[1,count]))

				count += 1 # increase the count everytime

			# Now check if the polygon has all vertices within the boundary.
			for i in polygon:
				# Set up a for loop to flag if the polygon goes beyond the boundary
				if not xMin <= i[0] <= xMax or not xMin <= i[1] <= yMax:
					polyInBoundary = False # Flag for if polygon is in boundary.
					break # Leave the loop if not in boundary.

			# Append the new polygon to the list if contains at least 3 points (not a dot or line).
			if polyInBoundary and len(polygon) >= 3:
				cells.append(cell.cell(polygon, polygon, None))
	
	return cells
	


	# def get_mags_angles(self):
	#
	# 	edges = self.get_edges()
	#
	# 	# Now make a list of the vectors representing the edges.
	# 	edgeVecs = [x[1] - x[0] for x in self.edges] # does first array minus second in every tuple.
	# 	# Now we want to remove all of the duplicate vectors that are the same, but just point in
	# 	# the other direction.
	# 	# Iterate through and remove the negative versions of the vectors.
	# 	[mymath.rmArray(edgeVecs, j) for i in edgeVecs for j in edgeVecs if -j[0] == i[0] and -j[1] == i[1]]
	#
	# 	# THE CODE BELOW DEALS WITH ANALYSIS OF THE CELL EDGES.
	# 	# Now calculate the angles using arctan:
	# 	divisors = [ x[1]/x[0] if x[0] != 0 else 0 for x in edgeVecs ]
	# 	self.angles = np.arctan(divisors) * 180 / np.pi
	# 	# edgeVecs_ = np.array(edgeVecs)
	# 	# angles = np.arctan2(edgeVecs_[:,0],edgeVecs_[:,1])* 180 / np.pi
	# 	self.angles = abs(self.angles)
	#
	# 	# Now get the magnitude of the vectors.
	# 	self.mags = [np.sqrt(i.dot(i)) for i in edgeVecs]
	# 	self.mags = np.asarray(self.mags) # Convert the list to an array for processing.
	# 	self.angles = np.asarray(self.angles) # Convert the list to an array for processing.
	#
	# 	return self.mags, self.angles
	#
	# #print [zip(*i)[0] for i in polygons] # Prints all x coords
	# #print [zip(*i)[1] for i in polygons] # Prints all y coords
	
	
