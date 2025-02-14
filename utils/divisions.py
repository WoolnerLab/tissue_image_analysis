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
    ## figure out how to unzip nuclei_rois, for now let nuclei_rois be the .roi file type
    ## later will be a .zip containing all rois, which will be iterated through

    nuclei_roi_files = os.listdir(nuclei_folder)
    dividing_cell_IDs = np.zeros(len(nuclei_roi_files))

    for nuclei in range(0,len(nuclei_roi_files)):
        #print("Looking for matches for nucleus: " + str(nuclei))
        roi = ImagejRoi.fromfile(nuclei_folder+nuclei_roi_files[nuclei])

        # first get centre and choose cells within specific radius to check 
        nucleus_centre = [np.mean(np.asarray(roi.coordinates())[:,0])*scale, np.mean(np.asarray(roi.coordinates())[:,1])*scale]

        ### threshold for cells within the nearest 100 microns
        distance_from_trace_cells = np.sqrt(((cell_centres - nucleus_centre)*(cell_centres - nucleus_centre)).sum(axis=1))
        ID_potential_matches = distance_from_trace_cells.argsort()[0:6][np.where(distance_from_trace_cells[distance_from_trace_cells.argsort()[0:6]]<neighbour_threshold)[0]] ### choose the top 6 
       
        if nuclei == 0:
            print("nucleus 0")
            plt.plot(np.asarray(roi.coordinates())[:,0]*scale, np.asarray(roi.coordinates())[:,1]*scale)
            for j in ID_potential_matches:
                plt.plot(R[np.append(cells[j], cells[j][0])][:,0],R[np.append(cells[j], cells[j][0])][:,1])
            plt.savefig("testing_cells.png",  dpi=300)

        ### now use ID_potential_matches to identify which polygon positions we're worrying about
        get_combined_areas = np.zeros(len(ID_potential_matches))
        get_cell_areas = np.zeros(len(ID_potential_matches))
        for i in range(0,len(ID_potential_matches)):
            get_combined_areas[i] = scipy.spatial.ConvexHull(np.vstack((R[cells[ID_potential_matches[i]]], np.asarray(roi.coordinates()*scale)))).area
            get_cell_areas[i] = scipy.spatial.ConvexHull(R[cells[ID_potential_matches[i]]]).area
        
        dividing_cell_IDs[nuclei] = ID_potential_matches[(get_combined_areas/get_cell_areas).argmin()]

    return dividing_cell_IDs
    #####