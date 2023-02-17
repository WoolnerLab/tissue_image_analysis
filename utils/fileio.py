"""
fileio.py

File input and output functions.

Created by Natasha Cowley on 2023-01-24.
"""
import glob
import os
import pandas as pd
import numpy as np
import csv
from datetime import datetime
from pyexcel.cookbook import merge_all_to_a_book

def setup_directories(output_dir, exp_ID):
    """ Set up directories to store outputs """
    if os.path.exists(output_dir+exp_ID)==False: os.mkdir(output_dir+exp_ID)

    mydir = os.path.join(output_dir+exp_ID, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
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

def read_conf(filename):
    """ read in config file """
    conf_data=pd.read_csv(filename)

    exp_date=str(conf_data.Exp_date[0])
    exp_ID=conf_data.Exp_ID[0]
    trace_type=int(conf_data.Trace_Type[0]) #0=Manual, 1=cellpose
    nuclei_exist=conf_data.Nuclei_Exist[0] #is there a nuclei file (probably not if in cellpose)
    edges_name=conf_data.Edges_Name[0]
    nuclei_name=conf_data.Nuclei_Name[0]
    stretch_type=int(conf_data.Stretch_Type[0]) # 0=Unstretched, 1=Fast Stretch, 2=Incremental Stretch
    t=conf_data.t_sec[0] #time in seconds of image
    pixel_size=conf_data.Pixel_Size[0] #from raw image
    micron_size=conf_data.Micron_Size[0] #from raw image

    return exp_date, exp_ID, trace_type, nuclei_exist, edges_name, nuclei_name, stretch_type,t, pixel_size, micron_size


def write_parameters(savedir,exp_ID,stretch_type,t,pixel_size, micron_size,Gamma, Lambda, pref_area, area_scaling_gradient):
    """write parameters used to file for reference"""
    if stretch_type==0 or t==0:
        area_scaling=0

    with open(savedir+'/'+exp_ID+'_parameters.txt', 'w') as f:
        f.write('# exp_ID,stretch_type,t,pixel_size, micron_size,Gamma, Lambda, pref_area unscaled, area_scaling_gradient \n')
        f.write('{}, {}, {},{}, {}, {},{}, {}, {} '.format(exp_ID,stretch_type,t,pixel_size, micron_size,Gamma, Lambda, pref_area, area_scaling_gradient))

def write_cell_data(savedir,cell_edges, cX, cY, eptm):
    """
    function to write the cell data from the trace to file.
     -Edges per cell
     -cell centroids
     -unique vertices and edges

    """
    #write edges per cell, non uniform number of edges per cell so slightly awkward.
    with open(savedir+'/edgesOnCells.csv', 'w', newline='') as output:
        writer = csv.writer(output)
        for key, value in sorted(cell_edges.items()):
            writer.writerow(value)
    merge_all_to_a_book(glob.glob(savedir+"/*.csv"), savedir+"/edgesOnCells.xlsx")
    #write cell centroids
    dataframe = pd.DataFrame({'cX':np.asarray(cX), 'cY':np.asarray(cY)})
    writer = pd.ExcelWriter(savedir+'/centroids.xlsx', engine='xlsxwriter')
    dataframe.to_excel(writer,sheet_name='samples')
    writer.close()

    #write vertices and edges
    np.save(savedir+"/uniqueTissueVerts.npy",np.asarray(eptm.global_vertices))
    np.save(savedir+"/UniqueTissueEdges.npy",eptm.global_edges)
    
def write_matrices(savedir,A, B, C,R):
    """
    Writes matrices A, B, C and vertex positions R to file 
    """
    np.savetxt(savedir+"/Matrix_A.txt",A)
    np.savetxt(savedir+"/Matrix_B.txt",B)
    np.savetxt(savedir+"/Matrix_R.txt",R)
    np.savetxt(savedir+"/Matrix_C.txt",C)

def write_pref_area(savedir,input_dir, exp_ID, pref_area):
    """ Write preffered area to file """
    with open(savedir+'/'+exp_ID+'_pref_area.txt', 'w') as f:
        f.write('{}'.format(pref_area))
    with open(input_dir+'/'+exp_ID+'_pref_area.txt', 'w') as f:
        f.write('{}'.format(pref_area))

def read_pref_area(savedir, exp_date):
    """ read preffered area from file """
    with open(glob.glob(savedir+"/"+exp_date+"*pref_area.txt")[0],"r") as f:
        pref_area=float(f.readline())
    return pref_area


def write_data(cell_data,savedir, exp_ID):
    """ write cell data to file """
    cell_data.to_csv(savedir + '/cell_data_'+exp_ID+'.csv', index=False)

def write_summary_stats(cell_data, savedir, exp_ID):
    """ calculate and write summary stats to file """
    cell_data.iloc[:,1:].describe().to_csv(savedir + '/summary_stats_'+exp_ID+'.csv')

def write_global_data(global_stress, total_energy, savedir, exp_ID):
    """write global quantities to file"""
    with open(savedir+'/'+exp_ID+'_global_data.txt', 'w') as f:
        f.write('# global_stress, monolayer_energy \n')
        f.write('{}, {}'.format(global_stress, total_energy))
