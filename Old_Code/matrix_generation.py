"""
Code to generate matrices from just trace data

"""

import os
import numpy as np
import pandas as pd

from datetime import datetime

from utils import fileio
from utils import matrices


CURRENT_DIR = os.getcwd()
input_dir=CURRENT_DIR+'/Input/'
output_dir=CURRENT_DIR+'/Output/'

#########################
#User Input
#########################

#Name of config file stored with input images
conf_file=input_dir+'conf_file.csv'

#Path to directory where Trace data are stored
data_dir='C:\\Users/v35431nc/Documents/Lab Stuff/Code/Natasha_Analysis_Code/Output/20191205_0min_Mycodetest/2023-02-06_15-52-06'

#########################
#Setup directories
#########################

#read in conf file
exp_date, exp_ID, trace_type, nuclei_exist, edges_name, nuclei_name, stretch_type,t_min, pixel_size, micron_size = fileio.read_conf(conf_file)

#make directories to output to
if os.path.exists(output_dir+exp_ID)==False: os.mkdir(output_dir+exp_ID)

mydir = os.path.join(output_dir+exp_ID, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.mkdir(mydir)
mat_dir=mydir+"/Matrices"
if os.path.exists(mat_dir)==False: os.mkdir(mat_dir)

#########################
#Load trace data
#########################

unique_edges = np.load(data_dir+"/UniqueTissueEdges.npy") #List of edges with vertex indices
unique_vertices = np.load(data_dir +"/uniqueTissueVerts.npy") #list of vertices with coords
sample = pd.read_excel(data_dir +"/centroids.xlsx" ) #used to get cell centres for clockwise stuff
cellEdges = pd.read_excel(data_dir +'/edgesOnCells.xlsx',header =None) #list of cells with edges ~ currently created by hand- ASK CHRIS

#########################
#Generate Matrices
#########################

cX = sample.cX.values
cY = sample.cY.values

N_c=len(cX)

cell_edges={}
for i in range(N_c):
    cell_edges[i]=np.asarray(cellEdges.iloc[i].dropna())

A, B, C, R = matrices.get_matrices(unique_edges,unique_vertices,cell_edges, cX, cY)

fileio.write_matrices(mat_dir, A, B, C, R)