"""
main.py
Natasha Cowley 2024/07/16


"""
import os
import numpy as np

from Glob import glob

from src import trace_processing
from src import geometry
from src import mechanics
from src import fileio
from src import visualise

#########################
#Set up directories
#########################

CURRENT_DIR = os.getcwd()
input_dir=CURRENT_DIR+'/Input/'
output_dir=CURRENT_DIR+'/Output/'

#########################
#User Input
#########################

trace_file='C:\\Users\\v35431nc\\Documents\\Lab_Stuff\\Parameter_Inference\\Traces/20231024_2_IP_GFPCAAX-CheHis_uu_0p5_MP_fr6_trace.tif'

#########################
#Constant variables
#########################


#########################
#Process trace and get matrices
#########################
R, A, B, C, G, cells, edge_verts=trace_processing.get_matrices(trace_file)

#########################
#Get geometric data
#########################

#make sure pixel size is size of tif not of trace as some have borders
R=R*(micron_size/pixel_size)

cell_areas=geometry.get_areas(A,B, R)
cell_perimeters=geometry.get_perimeters(A,B,R)
cell_edge_count=geometry.get_edge_count(B)
cell_centres=geometry.get_cell_centres(C,R,cell_edge_count)
mean_cell_area=geometry.get_mean_area(cell_areas)      
print("Mean area = ", mean_cell_area)

#########################
#Non-dimensionalisation
#########################


#########################
#Get mechanical data
#########################