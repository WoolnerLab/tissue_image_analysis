import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import  Polygon
from matplotlib.collections import PatchCollection

from datetime import datetime

from utils import fileio
from utils import handtrace
from utils import matrices
from utils import geometry
from utils import mechanics
from utils import visualise


def make_polygon(i, C, R, cell_centres):
    """
    Generate polygon

    Parameters:
    i (int): cell id
    C (numpy array): Nc x Nv order array relating cells to vertices
    R (numpy array): vertex coordinates
    cell_centres (numpy array): cell centre coordinates
    """

    Ralpha=R[np.where(C[i,:]==1)[0]]-cell_centres[i] #ref frame of cell
    ang=np.arctan2(Ralpha[:,1], Ralpha[:,0])%(2*np.pi) #find angle with x axis
    R_ang=np.transpose(np.vstack((np.where(C[i,:]==1)[0], ang))) #stack index of vertices with angle
    ordered_vertices=R_ang[np.argsort(R_ang[:,-1], axis=0)] #sort by anticlockwise angle
    polygon = Polygon(R[ordered_vertices[:,0].astype(int)],closed = True)
    return polygon

def plot_edges(A, R, color):
    N_e=np.shape(A)[0]
    beg_edge = ((abs(A) - A)*0.5)@R
    end_edge = ((abs(A) + A)*0.5)@R
    for j in range(0,N_e):
        plt.plot([beg_edge[j,0],end_edge[j,0]],[beg_edge[j,1],end_edge[j,1]],c=color,alpha=1.0,linestyle ='-', linewidth=1)

def plot_cell_centres(cell_centres, color):
    for i in range(len(cell_centres)):
        plt.plot(cell_centres[i,0],cell_centres[i,1],marker ='o',markersize=2, c=color)
        

def plot_polys(C, R, cell_centres):
    N_c=np.shape(C)[0]
    patches = []

    for i in range(N_c):
        polygon = make_polygon(i, C, R, cell_centres)
        patches.append(polygon)

    p = PatchCollection(patches,alpha = 1.0)
    return p

#establish directory structure
CURRENT_DIR = os.getcwd()
input_dir='C:\\Users\\v35431nc\\Documents\\Lab_Stuff\\Movies_to_track\\Incremental/20240417_1_IP_GFPCAAX-CheHis_us_8p6_SumP/Frames/aligned/traces/'
save_dir='C:\\Users\\v35431nc\\Documents\\Lab_Stuff\\Movies_to_track\\Incremental/20240417_1_IP_GFPCAAX-CheHis_us_8p6_SumP/Frames/aligned/traces/green_edges/'

px = 1/plt.rcParams['figure.dpi']  # pixel in inches

tr_files=sorted(glob.glob(input_dir+'*_trace.tif'))[:]
for f in tr_files:
    print(f)
    edges_name=f.split('\\')[-1].split('.')[0]
    exp_id=edges_name.split('_')[0]+'_'+edges_name.split('_')[1]+'_'+edges_name.split('_')[7]

    print("Extracing trace data")
    edge_verts,cells, cell_edges, A, B, C, R, image0 = handtrace.run_trace(edges_name, input_dir)
    #handtrace.check_trace_plot(save_dir,image0, cells, R, edges_name)

    cell_edge_count=geometry.get_edge_count(B)
    cell_centres=geometry.get_cell_centres(C,R,cell_edge_count)

    fig, ax = plt.subplots(frameon=False,figsize=(1080*px, 1080*px),subplot_kw={'aspect': 'equal'})

    R=R-np.mean(R, axis=0)
    ## For colormap of continuous data

    polys=plot_polys(C, R, cell_centres)
    polys.set_facecolor('black')

    ax.add_collection(polys) 

    ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    #ax.axis('off')
    ## Add edges to plot 
    plot_edges(A,R, 'green')

    plt.xlim(-540, 540)
    plt.ylim(-540,540)
    ax.set_facecolor("#000000")
    plt.gca().set_aspect('equal')
    plt.tight_layout()

    plt.savefig(save_dir+'/check_trace_'+edges_name+'.png', bbox_inches='tight', pad_inches=0)
    plt.close()
