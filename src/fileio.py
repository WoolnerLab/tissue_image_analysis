"""
fileio.py
Natasha Cowley 2024/07/16

Functions to handle input and output of the code.
"""

import glob
import os
import pandas as pd
import numpy as np
import csv
from datetime import datetime


def setup_directories(output_dir, edges_name):
    """ Set up directories to store outputs """
    if os.path.exists(output_dir+edges_name.split('_trace')[0])==False: os.mkdir(output_dir+edges_name.split('_trace')[0])

    mydir = os.path.join(output_dir+edges_name.split('_trace')[0], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(mydir)

    trdir=mydir+"/Trace_extraction"
    matdir=mydir+"/Matrices"
    datadir=mydir+"/Data"
    plotdir=mydir+"/Plots"

    if os.path.exists(trdir)==False: os.mkdir(trdir)
    if os.path.exists(matdir)==False: os.mkdir(matdir)
    if os.path.exists(datadir)==False: os.mkdir(datadir)
    if os.path.exists(plotdir)==False: os.mkdir(plotdir)
    
    return mydir, trdir, matdir, datadir, plotdir


def read_conf_line(line):
    "split line from conf file into info"
    conf_data=line.split(',')
    edges_name=conf_data[0] # file name excluding .tif
    t_min=float(conf_data[1]) #time in minutes of image
    pixel_size=float(conf_data[2]) #from raw image
    micron_size=float(conf_data[3]) #from raw image

    return  edges_name, t_min, pixel_size, micron_size

def write_parameters(savedir,edges_name,stretch_type,t,pixel_size, micron_size,Gamma, Lambda, pref_area, area_scaling_gradient):
    """write parameters used to file for reference"""
    if stretch_type=='u' or t==0:
        area_scaling=0

    with open(savedir+'/'+edges_name+'_parameters.txt', 'w') as f:
        f.write('# exp_ID,stretch_type,t_sec,pixel_size, micron_size,Gamma, Lambda, pref_area unscaled, area_scaling_gradient \n')
        f.write('{}, {}, {},{}, {}, {},{}, {}, {} '.format(edges_name,stretch_type,t,pixel_size, micron_size,Gamma, Lambda, pref_area, area_scaling_gradient))

def write_cell_data(savedir,edge_verts, cells, exp_id):
    """
    function to write the cell data from the trace to file.
     -vertices connected by an edge
     -vertices in cells
     -edges in cells

    """

    #write vertices per cell, non uniform number of verts per cell so slightly awkward.
    with open(savedir+'/'+exp_id+'_cell_vertices.csv', 'w', newline='') as output:
        writer = csv.writer(output)
        for i in cells:
            writer.writerow(i)
    #write vertices and edges
    np.savetxt(savedir+'/'+exp_id+"_edge_verts.csv",np.asarray(edge_verts))

    
def write_matrices(savedir,A, B, C,R, exp_id):
    """
    Writes matrices A, B, C and vertex positions R to file 
    """
    np.savetxt(savedir+'/'+exp_id+"_Matrix_A.txt",A)
    np.savetxt(savedir+'/'+exp_id+"_Matrix_B.txt",B)
    np.savetxt(savedir+'/'+exp_id+"_Matrix_R.txt",R)
    np.savetxt(savedir+'/'+exp_id+"_Matrix_C.txt",C)

def write_pref_area(savedir,input_dir, edges_name, pref_area):
    """ Write preffered area to file """
    with open(savedir+'/'+edges_name.split('_trace')[0]+'_pref_area.txt', 'w') as f:
        f.write('{}'.format(pref_area))
    with open(input_dir+'/'+edges_name.split('_trace')[0]+'_pref_area.txt', 'w') as f:
        f.write('{}'.format(pref_area))

def read_pref_area(savedir, edges_name):
    exp_date=edges_name.split('_')[0]+'_'+edges_name.split('_')[1]
    """ read preffered area from file """
    with open(glob.glob(savedir+"/"+exp_date+"*pref_area.txt")[0],"r") as f:
        pref_area=float(f.readline())
    return pref_area



def write_global_data(global_stress, total_energy, savedir, edges_name):
    """write global quantities to file"""
    with open(savedir+'/'+edges_name.split('_trace')[0]+'_global_data.txt', 'w') as f:
        f.write('# global_stress, monolayer_energy \n')
        f.write('{}, {}'.format(global_stress, total_energy))