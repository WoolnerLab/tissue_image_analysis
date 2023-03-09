from source import tissue
from source import cell
import cv2
import numpy as np
from source import setup_points
from source import graham_scan as gs
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
from mahotas.morph import hitmiss
from skimage.filters import gaussian
from skimage.feature import peak_local_max


########################################################################################################################
########################################################################################################################
########################################################################################################################


# 
# def skeleton(img):
#   """ Function to create skeleton image """
#     size = np.size(img)
#     skel = np.zeros(img.shape,np.uint8)
#
#     # ret,img = cv2.threshold(img,2,255,0)
#     # img = cv2.adaptiveBilateralFilter(img,(9,9),150)
#     # img = cv2.GaussianBlur(img,(5,5),0)
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
#     done = False
#
#     while( not done):
#         eroded = cv2.erode(img,element)
#         temp = cv2.dilate(eroded,element)
#         temp = cv2.subtract(img,temp)
#         skel = cv2.bitwise_or(skel,temp)
#         img = eroded.copy()
#
#         zeros = size - cv2.countNonZero(img)
#         if zeros==size:
#             done = True
#
#     return skel




def find_branch_points(skel):
    """ Function to identify trijunctions by looking from branch points in a skeleton image.
Returns branch points as x, coords. """
    xbranch0 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    xbranch1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    tbranch0 = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]])
    tbranch1 = np.flipud(tbranch0)
    tbranch2 = tbranch0.T
    tbranch3 = np.fliplr(tbranch2)
    tbranch4 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]])
    tbranch5 = np.flipud(tbranch4)
    tbranch6 = np.fliplr(tbranch4)
    tbranch7 = np.fliplr(tbranch5)
    ybranch0 = np.array([[1, 0, 1], [0, 1, 0], [2, 1, 2]])
    ybranch1 = np.flipud(ybranch0)
    ybranch2 = ybranch0.T
    ybranch3 = np.fliplr(ybranch2)
    ybranch4 = np.array([[0, 1, 2], [1, 1, 2], [2, 2, 1]])
    ybranch5 = np.flipud(ybranch4)
    ybranch6 = np.fliplr(ybranch4)
    ybranch7 = np.fliplr(ybranch5)
    block = np.array([[1, 1], [1, 1]])
    br1 = hitmiss(skel, xbranch0)
    br2 = hitmiss(skel, xbranch1)
    br3 = hitmiss(skel, tbranch0)
    br4 = hitmiss(skel, tbranch1)
    br5 = hitmiss(skel, tbranch2)
    br6 = hitmiss(skel, tbranch3)
    br7 = hitmiss(skel, tbranch4)
    br8 = hitmiss(skel, tbranch5)
    br9 = hitmiss(skel, tbranch6)
    br10 = hitmiss(skel, tbranch7)
    br11 = hitmiss(skel, ybranch0)
    br12 = hitmiss(skel, ybranch1)
    br13 = hitmiss(skel, ybranch2)
    br14 = hitmiss(skel, ybranch3)
    br15 = hitmiss(skel, ybranch4)
    br16 = hitmiss(skel, ybranch5)
    br17 = hitmiss(skel, ybranch6)
    br18 = hitmiss(skel, ybranch7)
    br19 = hitmiss(skel, block)

    # Collate all of the branches except the blob
    soFar = br1+br2+br3+br4+br5+br6+br7+br8+br9+br10+br11+br12+br13+br14+br15+br16+br17+br18  # +br19
    r, c = np.nonzero(soFar)  # Get their coords
    currentPoints = np.dstack([r, c])[0]  # as a list of points
    # Similarly for the blobs:
    rows, cols = np.nonzero(br19)  # get the coords of the blobs
    if len(rows) > 0:
        # as points (shifted up and left to centre on the blob)
        possPoints = np.dstack([rows-.5, cols-.5])[0]
        tree = cKDTree(possPoints)  # create a KDTtree
        # Loop over each junction and see if there is already a blob nearby. If not, store it.
        newRows, newCols = [], []
        for p in currentPoints:
            indices = tree.query_ball_point(p, 1.5)
            if len(indices) == 0:
                newRows.append(p[0]), newCols.append(p[1])
        # Reset the branches image and add the accepted ones
        soFar[:] = 0
        soFar[newRows, newCols] = 1
        # # return all of the junctions
        branches = soFar + br19  # Note, this is slightly different (blobs not shifted by 0.5)

        # rows,cols = np.nonzero(branches)
        # Concatenate the blobs and normal junctions and slightly shift the blobs to centre them
        return np.concatenate((cols-0.5, newCols)), np.concatenate((rows-0.5, newRows)), branches
    else:
        return c, r, soFar


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


########################################################################################################################
########################################################################################################################
########################################################################################################################


from scipy import ndimage as ndi
from skimage.morphology import dilation, skeletonize, disk
from skimage.segmentation import watershed
from skimage.feature import canny


def watershed_edges(nucleiFile, filename, holesFilename=None, maxCellSize=4000, minCellSize=50,
                    nucleusSize=5, smoothing=2, offSet=0):

    # Detect nuclei for seed points
    if nucleiFile is not None:
        seedsImage, nuclei, nucleiImage = setup_points.detect_nuclei(
            nucleiFile, holesFilename, nucleus_size=nucleusSize)

        # Initialise the seeds with their own unique colour.
        shape = seedsImage.shape
        seeds = np.zeros(shape)
        r = 0
        for x in range(0, shape[0]):
            for y in range(0, shape[1]):
                if seedsImage[x, y] == True:
                    r += 1
                    seeds[x, y] = r  # This just puts a  number at the x,y coord of a seed
        # Dilate the seeds so they work better
        markers = dilation(seeds, disk(smoothing))

        # Read the edges image
        edges, nx, ny, points = setup_points.read_edges_image(filename, smoothing=smoothing)
        # Pruning once we have a cell.
        pruneVal = 4
    else:
        smoothing = 1
        # Read the edges image
        edges, nx, ny, points = setup_points.read_edges_image(
            filename, smooth=False, smoothing=smoothing)
        pruneVal = 12

        # Make storage to hold the cells:
    cells = tissue.tissue()
    # plt.imshow(edges)
    # plt.show()

    # Get the trijunctions
    branchXs, branchYs, branchesImage = find_branch_points(edges)
    allTriJuncs_ = np.dstack([branchXs, branchYs])[0]
    # allTriJuncs_ = np.array(setup_points.fuse(allTriJuncs_, 1.5))
    juncsTree = cKDTree(allTriJuncs_)
    allTriJuncs = set(map(tuple, allTriJuncs_))
    #print(allTriJuncs)
    """ Find the watershed elements: """

    # Now we do the watershedding:
    # Get the distance image from the edges
    # edges = gaussian(edges, sigma=0.2)
    # from skimage.feature import peak_local_max
    distance = ndi.distance_transform_edt(edges)
    # If we haven't got a seed image, try to deduce one.
    if nucleiFile is None:
        local_maxi = peak_local_max(distance)  # , labels=edges1)
        markers = ndi.label(distance)[0]
    # distance = ndi.distance_transform_edt(skeletonize(dilation(edges, disk(2))))
    # Now get the regions:
    labels = watershed(distance, markers)
    # Now we loop over the individual regions
    count = 0
    for i in np.unique(labels.flatten()):
        # Make a copy of the watershed labels
        labels_copy = labels.copy()
        # Set all regions except the current one to zero.
        labels_copy[labels_copy != i] = 0
        labels_copy[labels_copy == i] = 1

        # See if any seed points are in labels_copy, if not, we dont use this segmentation
        if nucleiFile is not None:
            c = np.where(nucleiImage == labels_copy, nucleiImage, 0)
        else:
            c = np.array([1])

        # print all((j == nucleiImage).flatten())
        area = labels_copy[labels_copy > 0].size
        #print(i)
        #print(area)
        # If we find huge/tiny regions its probs not a cell
        if area > minCellSize and area < maxCellSize and any(c.flatten()):
            #print (count)
            # y,x = np.nonzero(labels_copy)
#             plt.plot(x,y,'o', ms=2)
#             plt.imshow(edges)
#             # plt.hold(True)
#             plt.show()

            # Get the polygon from the edge detection:
            # polygon = canny(labels_copy, sigma=1)
            # polygon = skeletonize(polygon) # SKELETONISE LATER>>> NEED THIICK TO FIND TRIJUNCTIONS

            ########################

            # ALTERNATIVELY: DILATE THE TRACE AND FIND THE OVERLAP WITH THE ORIGINAL IMAGE.

            # Get the interior of the polygon
            # Get the x,y coords
            y, x = np.nonzero(labels_copy)
            # Tuples:
            interior = np.dstack([x, y])[0] - offSet

            # plt.imshow(labels_copy)
            # plt.show()
            labels_copy = dilation(labels_copy, disk(1))
            # plt.imshow(labels_copy)
            # plt.show()
            polygon = np.where(labels_copy == edges, labels_copy, 0)
            # plt.imshow(polygon)
            # plt.show()
            # Prune edges from neighbouring cells
            polygon = pruning(skeletonize(polygon), pruneVal)

            ########################

            # Get the x,y coords
            y, x = np.nonzero(polygon)
            # Tuples:
            points = np.dstack([x, y])[0]

            # Now see if any tricellular junctions were within 1 pixel of the edges:
            nearJuncs = juncsTree.query_ball_point(points, 1)
            nearJuncs = np.unique([item for sublist in nearJuncs for item in sublist])
            if nearJuncs.size > 0:
                triJunctions = set(map(tuple, allTriJuncs_[nearJuncs]-offSet))

            ########################

            # # Now seletonise the edges detection and add trijunctions:
            # polygon = skeletonize(polygon)
            # y,x = np.nonzero(polygon)
            # points = np.dstack([x,y])[0]
            # # Merge the boundary with the trijunctions
            # points = set(map(tuple,points))
            # points = points | triJunctions # uninon of sets
            # # triJunctions = allTriJuncs.intersection(polygon)
            # points = np.array(list(points))
            # # Reduce the number of points needed to create the boundary.
            # polygon = cv2.approxPolyDP(points, 1, True)
            # # Append the cell to the sheet of cells.
            # cells.add_cell(cell.cell(polygon[:,0], triJunctions))

            # OR FIND THE CONTOUR OF THE LABEL USING OPENCV:
            # labels_copy = gaussian(labels_copy, sigma=0.2)
            # points, hierarchy = cv2.findContours(cv2.convertScaleAbs(labels_copy), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # points = set(map(tuple,points[0][:,0]))
            # # points = points | triJunctions # uninon of sets
            # polygon = np.array(list(points))
            # cells.add_cell(cell.cell(polygon, triJunctions))

            # OR FIND THE OULINE WITH SCIPY
            # segmentation = ndi.binary_fill_holes(labels_copy - 1)
            # polygon = skeletonize(polygon)
            # y,x = np.nonzero(polygon)
            # points = np.dstack([x,y])[0]
            # cells.add_cell(cell.cell(points, triJunctions))

            ########################

            points -= offSet
            points = set(map(tuple, points))
            points = points | triJunctions  # uninon of sets
            # Add the trijunctions and boundary to the interior
            inter = set([tuple(i) for i in interior])
            interior = np.array(list(inter | points))
            points = np.array(list(points))

            # polygon = cv2.approxPolyDP(points, 1, True)
            # cells.add_cell(cell.cell(polygon[:,0], triJunctions))
            cells.add_cell(cell.cell(points, triJunctions, interior))
            count += 1

    return cells


def get_cell_edges(nuclei, filename):

        # # Load the image.
        # img = cv2.imread(filename,0)
        #
        # # blur = cv2.bilateralFilter(img,9,75,75)
        # blur = cv2.adaptiveBilateralFilter(img,(9,9),150)
        # blur = cv2.GaussianBlur(blur,(5,5),0)
        #
        # # edges = cv2.Canny(blur, 50, 100, apertureSize = 3, L2gradient=True)
        #
        # edges = skeleton(blur)
        # # Remove spurious marks:
        # edges = cv2.GaussianBlur(edges,(3,3),0)
        # # edges[edges<10] = 0
        # # edges = blur
        # edges[edges<7] = 0

        # # Load the image.
        # img = cv2.imread(filename,0)
        # img[img < 10] = 0 # Remove spurious noise.
        #
        # # blur = cv2.bilateralFilter(img,9,75,75)
        # blur = cv2.adaptiveBilateralFilter(img,(9,9),150)
        # blur = cv2.GaussianBlur(blur,(5,5),0)
        #
        # # edges = cv2.Canny(blur, 50, 100, apertureSize = 3, L2gradient=True)
        #
        # edges = skeleton(blur)
        # edges[edges<20] = 0 # REDUCE THIS THRESHOLD IF THE IMAGE IS LEFT WITH GAPS. (20 GOOD)
        #
        # # Get the coordinates of the non-empty pixels.
        # y,x = np.nonzero(edges)
        # # Make tuples of the x,y coordinates.
        # points = zip(x,y)

    edges, nx, ny, points = setup_points.read_edges_image(filename)

    # Get the trijunctions
    branchXs, branchYs = find_branch_points(edges)
    allTriJuncs_ = np.dstack([branchXs, branchYs])[0]
    juncsTree = cKDTree(allTriJuncs_)
    allTriJuncs = set(map(tuple, allTriJuncs_))

    # allTriJuncs = set([ list(i) for i in allTriJuncs_ ])

    # edges = morphology.skeletonize(blur > 0)

    # plt.subplot(121),plt.imshow(blur,cmap = 'gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

    ########################################################################################################################

    # Make storage to hold the cells:
    cells = tissue.tissue()

    # Make a kd-tree of the points.
    tree = cKDTree(points)

    # Keep track of the number of cells:
    cellCount = 0

    for nucleus in nuclei:
        triJunctions = set([])
        # try:
        # Get the nearest point to the nucleus.
        d, i = tree.query(nucleus, 1)
        startPoint = points[i]
        # Initialise the last point added to the polygon.
        lastPoint = startPoint

        # Initialise the list of points.
        polygon = [tuple(startPoint)]

        # Initialise a checker to keep track of how many times we have moved 180 degrees.
        moved180 = 0
        # Initialise a boolean to check whether the next point it in an anticlockise direction relative to the start.
        anticlockwise = True

        # Loop over the points and add the correct points to the polygon.
        while moved180 < 2:  # startPoint != nextPoint:

            # This checks if we query the tree for enough points such that at least 1 goes anticlockwise.
            queriedEnough = False
            queryNumber = 4  # The number of points to return from querying the kdtree.
            # This counts how many of the points that were queried are in an anticlockwise direction.
            antiCount = 0

            while not queriedEnough:
                                # Add 2 to the query number each time the loop runs:
                queryNumber += 1
                # Find the closest points to the last point.
                dists, indices = tree.query(lastPoint, queryNumber)  # 8

                # Make a separate array consisting only of the nearest points.
                nearPoints = np.asarray(points)[indices]

                # Now loop over the points and check if any go antclockwise.
                for point in nearPoints:
                                        # If any do go anticlockwise, break out of the loop and proceed.
                    if gs.less(point, lastPoint, nucleus):
                        antiCount += 1

                        # If there are at least 4 anticlockwise points, break the loop.
                if antiCount > 4:
                    queriedEnough = True

                    # # Find the closest points to the last point.
                    # dists, indices =  tree.query(lastPoint, queryNumber) # 8
                    #
                    # # Make a separate array consisting only of the nearest points.
                    # nearPoints = np.asarray(points)[indices]

                    # Calculate the distances of the nearest points to the nucleus.
            distances = [np.sqrt((point[0] - nucleus[0])**2 + (point[1] - nucleus[1])**2)
                         for point in nearPoints]

            # Transpose nearpoints.
            nearPoints = nearPoints.T

            # Sort in order of distance from the nucleus.
            indexlist = sorted(range(len(distances)), key=distances.__getitem__)
            # indexlist.reverse()
            nearPoints = nearPoints[:, indexlist]
            nearPoints = zip(nearPoints[0, :], nearPoints[1, :])

            found = False
            count = 0

            # Cycle through the points until we find one that moves anticlockwise.
            while not found:

                if gs.less(nearPoints[count], lastPoint, nucleus):
                    nextPoint = nearPoints[count]
                    found = True

                else:
                    count += 1

                # Store the next point in the polygon list.
            polygon.append(nextPoint)
            nearJuncs = juncsTree.query_ball_point(nextPoint, 2)
            if len(nearJuncs) > 0:
                triJunctions.update(map(tuple, [allTriJuncs_[nearJuncs[0]]]))
            # Now also check if we missed a nearby junctions, by finding all nearby next and last point and
            # seeing if there is one in between them
            nearJuncsLast = set(juncsTree.query_ball_point(lastPoint, 2))
            nearJuncsNext = set(juncsTree.query_ball_point(nextPoint, 2))
            intersect = nearJuncsLast.intersection(nearJuncsNext)
            if intersect:
                triJunctions.update(map(tuple, [allTriJuncs_[list(intersect)[0]]]))

               # Update the last point
            lastPoint = nextPoint

            # Check if we are no longer moving clockwise/anticlockwise relative to the point.
            if gs.less(nextPoint, startPoint, nucleus) != anticlockwise:
                anticlockwise = gs.less(nextPoint, startPoint, nucleus)
                moved180 += 1

                # Update the user with the progress.
        cellCount += 1
        print (cellCount)

        # Fuse the trijunctions incase we got double hits. This combines ones that are really close
        # together.
        # triJunctions = set( setup_points.fuse(list(triJunctions), 2) )

        polygon = set(polygon)
        polygon = polygon | triJunctions  # uninon of sets
        # triJunctions = allTriJuncs.intersection(polygon)
        polygon = np.array(list(polygon))

        # # Append the cell to the sheet of cells.
        # cells.add_cell(cell.cell(polygon))

        # Reduce the number of points needed to create the boundary.
        polygon = cv2.approxPolyDP(polygon, 1, True)
        # Append the cell to the sheet of cells.
        cells.add_cell(cell.cell(polygon[:, 0], triJunctions))
        #

        # plt.plot(polygon.T[0], polygon.T[1], 'rs')
        # plt.plot(nucleus[0], nucleus[1], 'bs')
        # except:
        #     print cellCount, 'failed'

    return cells

    #     # If there are at least 2 anticlockwise points, break the loop.
    #     if antiCount > 3: queriedEnough = True
    #
    #
    # antiGroup = np.array(antiGroup).T
    #
    # distances = [ (point[0] - nucleus[0])**2 + (point[1] - nucleus[1])**2 for point in antiGroup ]
    #                 # Sort in order of distance from the nucleus.
    # indexlist = sorted(range(len(distances)), key=distances.__getitem__)
    # antiGroup = antiGroup[:,indexlist]
    # antiGroup = zip(antiGroup[0,:],antiGroup[1,:])
    # nextPoint = antiGroup[0]


# # WATERSHED ALGORITHM WITH MORE POSSIBILITIES
# def watershed_edges(nuclei, filename):
#
#     edges, nx, ny, points = setup_points.read_edges_image(filename)
#
#     # Get the trijunctions
#     branchXs, branchYs = find_branch_points(edges)
#     allTriJuncs_ = np.dstack([branchXs,branchYs])[0]
#     juncsTree = cKDTree(allTriJuncs_)
#     allTriJuncs = set(map(tuple, allTriJuncs_))
#
#     # Get Find the watershed elements
#     from scipy import ndimage as ndi
#
#     from skimage.morphology import watershed, dilation, convex_hull_object, disk
#     from skimage.feature import peak_local_max
#     from skimage.segmentation import find_boundaries, mark_boundaries
#
#     # edges1 = dilation(edges1, disk(2))
#     distance = ndi.distance_transform_edt(edges1)
#     local_maxi = peak_local_max(distance)#, labels=edges1)
#     markers = ndi.label(distance)[0]
#
#     import random
#     # im1[im1 < 1] = 0
#     shape = im1.shape
#     result = np.zeros(shape)
#     r=0
#     for x in range(0, shape[0]):
#         for y in range(0, shape[1]):
#             if im1[x, y] == True:
#                 r+=1
#                 result[x, y] = r
#             elif im1[x,y] == False: im1[x,y]=0
#
#     # labels = watershed(distance, markers)
#     markers = dilation(result, disk(3))
#     labels = watershed(distance, markers)
#     # bounds = find_boundaries(labels)#,mode='outer')
#     # plt.imshow(bounds, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
#     # plt.imshow(labels)
#     # plt.show()
#     from skimage import color, feature
#     import skimage.morphology as sk
#     print len(np.unique(labels.flatten()))
#     count = 0
#     for i in np.unique(labels.flatten()):
#         if count != 0:
#             ax = plt.subplot(1,1,1, axisbg='white')
#             labels_copy = labels.copy()
#             labels_copy[labels_copy != i] = 0
#             color_labels = color.label2rgb(labels_copy, edges1)
#             ax.imshow(color_labels[:300, :300])
#             ax.imshow(edges1)
#             # plt.show()
#             # labels_copy[labels_copy>0] = True
#             # labels_copy[labels_copy<1] = False
#
#             polygon = feature.canny(labels_copy, sigma=1)
#             polygon = sk.skeletonize(polygon)
#             y,x = np.nonzero(polygon)
#             ax.plot(x,y, 'o', c='yellow')#, cmap=plt.cm.gray)
#             # ax.imshow(color_labels[:300, :300])
#
#             #             y,x = np.nonzero(labels_copy)
#             # # Zip the points into tuples.
#             #             points = np.dstack([x,y])[0]
#             #             concave_hull, edge_points = alpha_shape(points, alpha=.4)
#             #             # print concave_hull
#             #             x,y = concave_hull.exterior.xy
#             #             plt.plot(x,y)
#
#             plt.show()
#         count += 1
