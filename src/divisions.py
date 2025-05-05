"""
divisions.py

Functions to assign dividing cells to population traces

Dividing cells identified externally in Fiji, using Stardist2D.
Reads in nuclei ROI and determines which cell is dividing
    """

import os
import numpy as np
import cv2
from roifile import ImagejRoi
import scipy.spatial 
import matplotlib.pyplot as plt

def get_dividing_cells(R, cells, cell_centres, nuclei_folder, scale, neighbour_threshold=100.0):
    """
   Return IDs of dividing cells.
   Take the ROIs of dividing nuclei and assign to the correct cell polygon, by checking the nearest 6 cells.

   R = positions of all vertices associated with each cell polygon
   cells = all cells 
   cell_centres = centre of cell polygons
   nuclei_folder = path to folder containing dividing nuclei ROIs
   scale = scaling between pixels and microns
   neighbour_threshold = maximum radius of checking for potential `host' cells. Defines a threshold for a guaranteed neighbouring cell.
    - will check fewer than the nearest 6 cells if one is deemed too far away to be a potential candidate.

    """

    nuclei_roi_files = os.listdir(nuclei_folder) # files containing dividing nuclei ROIs
    dividing_cell_IDs = np.zeros(len(nuclei_roi_files)) # pre-allocate array, each entry = 1 dividing cell

    for nuclei in range(0,len(nuclei_roi_files)): # for each nucleus... (find matching cell)
        roi = ImagejRoi.fromfile(nuclei_folder+nuclei_roi_files[nuclei])

        #get nucleus centre and choose cells within neighbour_threshold
        nucleus_centre = [np.mean(np.asarray(roi.coordinates())[:,0])*scale, np.mean(np.asarray(roi.coordinates())[:,1])*scale]

        ### threshold for nearest 6 cells, must lie within the nearest 100 microns
        distance_from_trace_cells = np.sqrt(((cell_centres - nucleus_centre)*(cell_centres - nucleus_centre)).sum(axis=1)) # distance between nucleus centre and all cell centres
        ID_potential_matches = distance_from_trace_cells.argsort()[0:6][np.where(distance_from_trace_cells[distance_from_trace_cells.argsort()[0:6]]<neighbour_threshold)[0]] ### choose the top 6 
       
       # test nucleus trace in nearest cells, example
        #if nuclei == 0:
        #    print("nucleus 0")
        #    plt.plot(np.asarray(roi.coordinates())[:,0]*scale, np.asarray(roi.coordinates())[:,1]*scale)
        #    for j in ID_potential_matches:
        #        plt.plot(R[np.append(cells[j], cells[j][0])][:,0],R[np.append(cells[j], cells[j][0])][:,1])
        #    plt.savefig("testing_cells.png",  dpi=300)

        ### now use ID_potential_matches to identify which polygon positions we're worrying about
        get_combined_areas = np.zeros(len(ID_potential_matches)) # hold area of convex hull enveloping nucleus + candidate cells
        get_cell_areas = np.zeros(len(ID_potential_matches)) # hold area of candidate cells
        for i in range(0,len(ID_potential_matches)): # for each candidate cell...
            get_combined_areas[i] = scipy.spatial.ConvexHull(np.vstack((R[cells[ID_potential_matches[i]]], np.asarray(roi.coordinates()*scale)))).area
            get_cell_areas[i] = scipy.spatial.ConvexHull(R[cells[ID_potential_matches[i]]]).area
        
        dividing_cell_IDs[nuclei] = ID_potential_matches[(get_combined_areas/get_cell_areas).argmin()] # find minumum of area ratios

    return dividing_cell_IDs
    #####