
""" Tissue class, which is a group of cells """


import numpy as np
from source import cell


class tissue(object):

    """ Initialiser: """

    def __init__(self):

        self.numberOfCells = 0

        # Make a container to store the cells.
        self.cells = []

        self.lineTension = -0.259
        self.contractility = 0.172

        """ Enable an iterator """

    def __iter__(self):
        for x in self.cells:
            yield x

    """ Set up len() to return number of cells """

    def __len__(self):

        return len([0 for i in self])

        # def next(self):
        # 	if not self.cells:
        # 		raise StopIteration
        # 	return self.cells.pop()

        """ Enable indexing: """

    def __getitem__(self, key):

        return self.cells[key]

        """ Enable setting of items """

    def __setitem__(self, key, value):
        self.cells[key] = value

        """ Enable deletion of a cell """

    def __delitem__(self, key):

        self.numberOfCells -= 1

        return self.cells[key]

        """ Method to add a cell to the tissue. """

    def add_cell(self, inCell):

                # Make a test cell to check if the input type is a cell.
        typeCheck = cell.cell([1, 2], set([]), [])

        # Check the datatype.
        if type(inCell) != type(typeCheck):
            print ('WANRING: You are adding something that is not a cell to the tissue')

            # Proceed if ok.
        else:
            self.cells.extend([inCell])

            self.numberOfCells += 1

    

    def update_geometry(self, shapeMethod='junctions'):

        """ Function to update the geometric data of the cells. """

        self.get_centroids()
        self.get_edges()
        self.get_areas()
        self.get_perims()
        self.get_evals_evecs(method=shapeMethod)


    def update_neighbours(self):

        """ Function to see if cells share triJunctions: If they do, then they are neighbours.
        Also, if a cell has more trijunctions than neighbours, then it is a boundary cell. """

        # Loop over each cell, checking its neighbours.
        for cell in self.cells:
            # Container for neighbours:
            naboer = set([])

            notDone = True  # This keeps us in a loop until there are no faulty junctions left.
            while notDone:
                # Trijuncs:
                triJuncs = cell.triJunctions
                goodJuncs = set([])
                possBadJuncs = []
                tempNaboer = []
                possNaboer = set([])
                # Loop over every cell in the tissue and get the neighbours.
                for possNabo in self.cells:
                    if possNabo != cell:
                        # Get the shared junctions
                        sharedJunctions = triJuncs.intersection(possNabo.triJunctions)
                        # If we have any shared junctions, store the cell for later:

                        if bool(sharedJunctions):
                            possNaboer.update([possNabo])
                        # Flag if we have > 2 neighbours.
                        if len(sharedJunctions) > 2:
                            possBadJuncs.append(sharedJunctions)

                if len(possBadJuncs) > 0:
                    badJuncs = possBadJuncs[0]
                    for i in possBadJuncs:
                        badJuncs = badJuncs.intersection(i)
                else:
                    badJuncs = set([])
                # Now we have a list of actual bad candidates, dicard the one furthest away for each neighbour.
                if bool(badJuncs):
                    # centroid = [np.mean([zip(*cell.vertices)[0]]),
                    #             np.mean([zip(*cell.vertices)[1]])]
                    centroid = [np.mean([list(zip(*cell.vertices))[0]]),
                                np.mean([list(zip(*cell.vertices))[1]])]
                    print (centroid)
                    # Loop over the cells ang see if all shared junctions are flagged
                    for nabo in possNaboer:
                        sharedJunctions = triJuncs.intersection(nabo.triJunctions)
                        if sharedJunctions.issubset(badJuncs):
                            sharedList = np.array(list(sharedJunctions))
                            if len(sharedList) > 1:
                                # Get the distances from the centroid to the triJunctions
                                dists = [sum((centroid - i)**2) for i in sharedList]
                                # Sort the trijunctions in order of how close they are
                                sharedList = sharedList[np.argsort(dists)]
                                cell.triJunctions = set(
                                    [j for j in triJuncs if j != tuple(sharedList[1])])
                                nabo.triJunctions = set(
                                    [j for j in nabo.triJunctions if j != tuple(sharedList[0])])
                                cell.vertices = [i for i in cell.vertices if tuple(
                                    i) != tuple(sharedList[1])]
                                nabo.vertices = [i for i in nabo.vertices if tuple(
                                    i) != tuple(sharedList[0])]
                            else:
                                cell.triJunctions = set([j for j in triJuncs])
                                nabo.triJunctions = set([j for j in nabo.triJunctions])

                    # if len(badJuncs) < 3: notDone = False
                else:
                    # naboer = possNaboer
                    notDone = False

            # Now that we have the neighbours, save them.
            cell.neighbours = possNaboer

            # If num neighbours is less than num triJuncs, mark as boundaryCell.
            if len(cell.neighbours) != len(cell.triJunctions):
                cell.isBoundaryCell = True

        # Update the geometry
        self.update_geometry()

    """ Function to derive a0, which nondimensionalises the experimental data """

    def set_a0(self, a0=2000):

        # Get the data:
        l, g = self.lineTension, self.contractility
        areas = np.array(self.get_areas(polygonise=True))
        perims = np.array(self.get_perims(polygonise=True))
        a0 = np.array([a0]*areas.size)

        l0 = -l/(2*g)

        oldSum = 100000000000000
        found = False
        while not found:
            areasNonDim = areas / a0
            perimsNonDim = perims / np.sqrt(a0)
            pEffs = areasNonDim - 1 + (g*perimsNonDim*(perimsNonDim - l0))/(2*areasNonDim)
            Area_Weighted_Peff = pEffs * areasNonDim
            globalStress = Area_Weighted_Peff.sum() / sum(areasNonDim)
            # print globalStress
            if globalStress > 0:
                oldSum = globalStress
                a0 += 1
            else:
                found = True

        self.a0 = a0[0]

        return self.a0

        """ Function to return all of the centroids """

    def get_centroids(self, method='junctions', polygonise=True):

                # self.centroids = []

        self.centroids = [cell.get_centroid(
            method=method, polygonise=polygonise) for cell in self.cells]

        return self.centroids

        """ Function to get edges. 'method' cand be 'junctions' or 'pixels' """

    def get_edges(self, polygonise=True):

        self.edges = []

        [self.edges.extend(cell.get_edges(polygonise)) for cell in self.cells]

        return self.edges

        """ Function to normalise distances relative to cell area. """

    def iso_scale(self):

        meanArea = np.mean(self.get_areas())

        for cell in self.cells:
            xys = zip(*cell.vertices)/np.sqrt(meanArea)
            cell.vertices = zip(xys[0], xys[1])

        """ Get the eigenvalues and vectors of each cell """

    def get_evals_evecs(self, method='junctions', polygonise=True):

                # Initialise the lists to hold the eigenvalues/vecs.
        self.eVals, self.eVecs, self.angles = [], [], []

        # Loop over every polygon.
        for poly in self.cells:

                        # Get the values/vecs for this polygon:
            if method == 'pixels':
                tVals, tVecs, angle = poly.get_evals_evecs()
            elif method == 'area':
                tVals, tVecs, angle = poly.get_evals_evecs_area(polygonise)
            elif method == 'perim':
                tVals, tVecs, angle = poly.get_evals_evecs_perim(polygonise)
            elif method == 'junctions':
                tVals, tVecs, angle = poly.get_evals_evecs_junctions()
            else:
                print ('ERROR NEED TO SPECIFY CORRECT SHAPE TENSOR')

                # Add the eigenvals/vecs to the big list.
            self.eVals.append(tVals)
            self.eVecs.append(tVecs)
            self.angles.append(angle)

        return self.eVals, self.eVecs, self.angles

        """ Get the principal Forces. """

    def get_forces(self):

        self.iso_scale()

        # Initialise the lists to hold the eigenvalues/vecs.
        self.eVals, self.eVecs, self.angles = [], [], []

        # Loop over every polygon.
        for poly in self.cells:

                        # Get the values/vecs for this polygon:
            tVals, tVecs, angle = poly.get_forces()

            # Add the eigenvals/vecs to the big list.
            self.eVals.append(tVals)
            self.eVecs.append(tVecs)
            self.angles.append(angle)

        return self.eVals, self.eVecs, self.angles

        """ Get the angles of the principal axes for every cell relative to the x axis """

    def get_axes_angles(self):

                # Initialise the list of angles.
        self.axesAngles = [cell.get_axis_angle() for cell in self.cells]
        #
        # for poly in self.cells:
        #
        # 	# Get the second order moments.
        # 	angle = poly.get_axis_angle()
        #
        # 	self.axesAngles.append(angle)

        # print self.axesAngles

        return self.axesAngles

        """ Function to return area of cells """

    def get_areas(self, polygonise=True):

        self.areas = [cell.get_area(polygonise) for cell in self.cells]

        return self.areas

    def get_areas_interior(self):

        self.areas = [cell.get_area_interior() for cell in self.cells]

        return self.areas

        """ Function to return perimeter of cells """

    def get_perims(self, polygonise=True):

        self.perimeters = [cell.get_perimeter(polygonise) for cell in self.cells]

        return self.perimeters

    def get_circularities(self, method='junctions', polygonise=True):

        self.circularities = np.array([cell.get_circularity(
            method=method, polygonise=polygonise) for cell in self.cells])

        return self.circularities

        """ Plot the Vertices of the cells """

    def plot_vertices(self):

        [i.plot_vertices() for i in self.cells]

        """ Plot principal axes of every cell. """

    def plot_pAxes(self, method='junctions', polygonise=True):

        [poly.plot_pAxes(method, polygonise) for poly in self.cells]

        """ Function to plot vertices """

    def plot_vertices(self):

        [i.plot_vertices() for i in self.cells]
