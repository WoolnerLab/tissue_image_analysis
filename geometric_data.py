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
from utils import handtrace
from utils import matrices
from utils import geometry
from utils import mechanics
from utils import visualise

#establish directory structure
CURRENT_DIR = os.getcwd()
input_dir=CURRENT_DIR+'/Input/'
output_dir=CURRENT_DIR+'/Output/CdhFL/'


files=sorted(glob.glob(input_dir+'20161130_2_GG_CadFL-GFPtub-CheHis_uf_8p6*conf.csv'))
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
    edges_name,t_min, pixel_size, micron_size = fileio.read_conf(conf_file)
    exp_id=edges_name.split('_')[0]+'_'+edges_name.split('_')[1]+'_'+edges_name.split('_')[7]
    t=t_min*60.0
    stretch_type=edges_name.split('_')[4][-1]
    #make directories to output to

    save_dir, trace_dir, matrix_dir, data_dir, plot_dir = fileio.setup_directories(output_dir, edges_name)

    ##############################
    #Run traces and construct matrices
    ##############################
    print("Extracing trace data")
    edge_verts,cells, cell_edges, A, B, C, R, image0 = handtrace.run_trace(edges_name, input_dir)
    fileio.write_cell_data(trace_dir,edge_verts,cells, cell_edges, exp_id)
    handtrace.check_trace_plot(save_dir,image0, cells, R, edges_name)
    print("Writing Matrices")
    fileio.write_matrices(matrix_dir, A, B, C, R, exp_id)

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
    #cell_circularity, major_axis, major_axis_alignment=geometry.get_shape_tensor(R,C,cell_edge_count,cell_centres)
    cell_circularity, major_shape_axis, major_shape_axis_alignment=geometry.get_shape_tensor(R,C,cell_edge_count,cell_centres, [])
    shape_parameter = cell_perimeters/(np.sqrt(cell_areas))


    geom_data_names=['cell_id', 'cell_perimeter_microns', 'cell_area_microns', 'shape_parameter', 'circularity', 'cell_edge_count', \
    'major_shape_axis_alignment_rads']
    cell_data_geom=np.transpose(np.vstack((cell_id, cell_perimeters, cell_areas,\
    shape_parameter, cell_circularity, cell_edge_count, major_shape_axis_alignment)))

    geom_df=pd.DataFrame(cell_data_geom, columns=geom_data_names)
    geom_df.to_csv(data_dir + '/'+exp_id+'_cell_data_geometry.csv', index=False)
    geom_df.iloc[:,1:].describe().to_csv(data_dir + '/'+exp_id+'_geometry_summary_stats.csv')



    ##############################
    #Plot
    ##############################
    print("Generating Plots")
    ### Distributions ###      

    visualise.plot_summary_hist(geom_df,'geom_data', plot_dir, edges_name)

    #discrete data
    visualise.plot_cell_sides(geom_df, "Number_of_Sides", plot_dir, edges_name)

    #angle histogram
    visualise.angle_hist(geom_df['major_shape_axis_alignment_rads'], "Major Shape Axis Alignment", plot_dir, 12, 180 , edges_name)
    axisLength = micron_size + 0.5
    visualise.graphNetworkColourBinary('Cell_id',geom_df['cell_id'],'black','black',0.0,0,0,t,A,C,R,cell_centres,cell_areas,major_shape_axis,axisLength,plot_dir,edges_name,ExperimentFlag, 1, 'blue','png')