"""
Created on 20/01/2023
Trace analysis



@author: Natasha Cowley
Contributors:
Emma Johns
Alex Nestor-Bergmann - image detection code
"""
import glob
import os

import numpy as np
import pandas as pd

from datetime import datetime


from source import tissue
from source import cell
from source import edge_detection
from source import setup_points
from source import graham_scan

from utils import fileio
from utils import trace
from utils import matrices
from utils import geometry
from utils import mechanics
from utils import visualise

#establish directory structure
CURRENT_DIR = os.getcwd()
input_dir=CURRENT_DIR+'/Input/'
output_dir=CURRENT_DIR+'/Output/Unstretched_Geometry/'

files=sorted(glob.glob(input_dir+'20200819_*.csv'))
for file in files:
    #########################
    #User Input
    #########################

    conf_file=file #name of config file

    #########################
    #Constant variables
    #########################

    Lambda = -0.259 # Line tension (tunes P_0) (non-dimensional)
    Gamma = 0.172  # Contractility (non-dimensional)
    pref_perimeter  = -Lambda/(2*Gamma)  # Cell preferred perimeter (non-dimensional)
    area_scaling = 0.00001785     # Gradient of the change in prefArea, used for time scaling. NB: if running code for knock downs you may need to recalculate this number.
    t_relax = 20.0 #computational parameter not important for image analysis                                  # not needed, take out if want to
    ExperimentFlag = 1   

    #########################
    #Setup directories
    #########################

    #read in conf file
    exp_date, exp_ID, trace_type, nuclei_exist, edges_name, nuclei_name, stretch_type,t, pixel_size, micron_size = fileio.read_conf(conf_file)

    #make directories to output to

    save_dir, trace_dir, matrix_dir, data_dir, plot_dir = fileio.setup_directories(output_dir, exp_ID)

    ##############################
    #Run traces
    ##############################
    print("Exp_ID = ", exp_ID)
    print("Extracing trace data")
    edgeToCell, centroidsX, centroidsY,  eptm, edgeNumber, cellEdges=trace.run_trace(edges_name, nuclei_name,nuclei_exist, input_dir)
    fileio.write_cell_data(trace_dir,cellEdges, centroidsX, centroidsY, eptm)
    trace.check_trace_plot(save_dir, eptm)

    ##############################
    #Construct Matrices
    ##############################
    print("Constructing Matrices")
    unique_edges = np.asarray(eptm.global_edges) #List of edges with vertex indices
    unique_vertices = np.asarray(eptm.global_vertices) #list of vertices with coords
    cX = np.asarray(centroidsX)
    cY = np.asarray(centroidsY)

    A, B, C, R = matrices.get_matrices(unique_edges,unique_vertices,cellEdges, cX, cY)

    fileio.write_matrices(matrix_dir, A, B, C, R)

    ##############################
    #Get cell geometry
    ##############################
    R=R*(micron_size/pixel_size)

    cell_areas=geometry.get_areas(A,B, R)
    cell_perimeters=geometry.get_perimeters(A,B,R)
    cell_edge_count=geometry.get_edge_count(B)
    cell_centres=geometry.get_cell_centres(C,R,cell_edge_count)
    mean_cell_area=geometry.get_mean_area(cell_areas)      
    print("Mean area = ", mean_cell_area)


    tangents=geometry.get_tangents(A,R)
    edge_lengths=geometry.get_edge_lengths(tangents)
    cell_centres=geometry.get_cell_centres(C,R,cell_edge_count)


    ##############################
    #Generate data
    ##############################
    print("Generating Data")

    N_c=np.shape(C)[0]
    edges = np.ma.masked_equal((abs(B)*edge_lengths), 0.0, copy=False)
    min_edge_length=edges.min(axis=1).data
    max_edge_length=edges.max(axis=1).data
    mean_edge_length=edges.mean(axis=1).data
    cell_id=np.linspace(0, N_c-1,N_c)
    cell_circularity, major_axis, major_axis_alignment=geometry.get_shape_tensor(R,C,cell_edge_count,cell_centres)
    shape_parameter = cell_perimeters/(np.sqrt(cell_areas))


    #combine data into a dataframe object
    data_names=['cell_id', 'cell_perimeter_microns', 'cell_area_microns', 'shape_parameter', 'circularity', 'cell_edge_count','major_axis_alignment_rads','max_edge_length', 'min_edge_length', 'mean_edge_length']
    cell_data=np.transpose(np.vstack((cell_id, cell_perimeters, cell_areas, shape_parameter, \
                cell_circularity, cell_edge_count, major_axis_alignment,max_edge_length, min_edge_length, mean_edge_length )))

    cell_df=pd.DataFrame(cell_data, columns=data_names) 

    fileio.write_data(cell_df, data_dir, exp_ID) #write calculated data

    fileio.write_summary_stats(cell_df, data_dir, exp_ID) #write summary stats



    ##############################
    #Plot
    ##############################
    print("Generating Plots")
    ### Distributions ###

    #Summary Histograms for continuous data 
    visualise.plot_summary_hist(cell_df, plot_dir, exp_ID)

    #discrete data
    visualise.plot_cell_sides(cell_df, "Number_of_Sides", plot_dir, exp_ID)

    #angle histogram
    visualise.angle_hist(cell_df['major_axis_alignment_rads'], "Major Axis Alignment", plot_dir, 6, 90 , exp_ID)


    visualise.graphNetworkColourBinary('Cell Elongation Binary',1.0-cell_circularity,'crimson','pink',2.0/3.0,1,0,\
    t,A,C,R,cell_centres,np.ones(N_c),major_axis,micron_size,plot_dir,exp_ID,ExperimentFlag, 'png')
