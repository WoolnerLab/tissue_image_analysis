from glob import glob
import pandas as pd
import imageio as iio
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import  Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import itertools
from skan.csr import skeleton_to_csgraph, make_degree_image, pixel_graph
from skan import Skeleton, summarize, draw
from skan.pre import threshold
from skimage import io
import skimage.morphology as sk
from skimage.segmentation import find_boundaries
from skimage.filters import rank
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import os

from datetime import datetime

from utils import fileio
from utils import handtrace
from utils import matrices
from utils import geometry
from utils import mechanics
from utils import visualise

from source import segmentation_hand


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
        plt.plot([beg_edge[j,0],end_edge[j,0]],[beg_edge[j,1],end_edge[j,1]],c=color,alpha=1.0,linestyle ='-')

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


folder='C:\\Users\\v35431nc\\Documents\\Lab_Stuff\\Movies_to_track\\unstretched\\20231019_1_IP_GFPCAAX-CheHis_uu_0p5_SumP\\5min_int'
files = glob(folder+'\\*_trace.tif')
#cp_cyto=np.load(files[0], allow_pickle=True)
#trace=find_boundaries(cp_cyto.item()['masks'])

for f in files:
    print(f)
    image0_cp =io.imread(f, as_gray=True)
    #skeletonise and segment edges
    trace_name=f.split('\\')[-1].split('.')[0]
    im_cp, data_cp=segmentation_hand.skeletonise(f)
  
    # draw.overlay_euclidean_skeleton_2d(im_cp,data_cp,
    #                                 skeleton_color_source='branch-type', skeleton_colormap='Set2');
    # plt.show()
    edge_verts_cp, n_coords_cp=segmentation_hand.extract_edges_verts(data_cp)
    Nv_cp=len(n_coords_cp)
    Ne_cp=len(edge_verts_cp)
    A_cp=segmentation_hand.construct_ev_incidence_matrix(edge_verts_cp, Ne_cp, Nv_cp)
    G_cp=segmentation_hand.construct_adjacency_matrix(A_cp)
    cells_cp=segmentation_hand.get_cycles(G_cp, 11)
    cell_edges_cp=segmentation_hand.assign_edges_to_cells(cells_cp, edge_verts_cp, Ne_cp)

    #For cellpose traces we need to remove the boundary cells
    edge_cells_cp=segmentation_hand.get_edge_cells(cells_cp, edge_verts_cp,Ne_cp)
    int_cells_cp,int_cell_edges_cp, int_edge_verts_cp, int_n_coords_cp=segmentation_hand.get_interior_cells(cells_cp, edge_cells_cp, cell_edges_cp, edge_verts_cp, n_coords_cp)

    #check connectivity

    Nc_cp=len(int_cells_cp)
    print("Nc = ", Nc_cp)
    Ne_cp=len(int_edge_verts_cp)
    Nv_cp=len(int_n_coords_cp)
    print(2+Ne_cp-Nv_cp-1)

    #get matrices

    A_cp=segmentation_hand.construct_ev_incidence_matrix(int_edge_verts_cp, Ne_cp, Nv_cp)
    B_cp=segmentation_hand.construct_ce_incidence_matrix(int_cells_cp,int_edge_verts_cp,int_cell_edges_cp, Nc_cp, Ne_cp)

    if len(np.where(B_cp@A_cp!=0)[0])!=0: print("Matrix generation error, check A and B")
        
    C_cp=0.5*(abs(B_cp)@abs(A_cp))

    R_cp=np.transpose(np.vstack((int_n_coords_cp[:,1],int_n_coords_cp[:,0])))

    #plot extracted cells
    fig, ax = plt.subplots()
    ax.imshow(im_cp, cmap='gray');
    for i in range(len(int_cells_cp)):
        ax.plot(int_n_coords_cp[np.append(int_cells_cp[i], int_cells_cp[i][0])][:,1],int_n_coords_cp[np.append(int_cells_cp[i], int_cells_cp[i][0])][:,0])
    plt.savefig(folder+'/segmentation_cells_'+trace_name+'.png', bbox_inches="tight", dpi=300)
    #plt.show()

    #Save matrices
    savedir=folder+'/Matrices'
    if os.path.exists(savedir)==False: os.mkdir(savedir)

    np.savetxt(savedir+"/Matrix_A_"+trace_name+".txt",A_cp)
    np.savetxt(savedir+"/Matrix_B_"+trace_name+".txt",B_cp)
    np.savetxt(savedir+"/Matrix_R_"+trace_name+".txt",R_cp)
    np.savetxt(savedir+"/Matrix_C_"+trace_name+".txt",C_cp)

    cell_edge_count=geometry.get_edge_count(B_cp)
    cell_centres=geometry.get_cell_centres(C_cp,R_cp,cell_edge_count)

    greendir=folder+'/green_edges'
    if os.path.exists(greendir)==False: os.mkdir(greendir)

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    R_cp=R_cp-np.mean(R_cp, axis=0)
    ## For colormap of continuous data

    polys=plot_polys(C_cp, R_cp, cell_centres)
    polys.set_facecolor('black')

    ax.add_collection(polys) 

    ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)

    ## Add edges to plot 
    plot_edges(A_cp,R_cp, 'green')

    #plt.xlim(0, 600)
    #plt.ylim(0,600)
    ax.set_facecolor("#000000")
    plt.gca().set_aspect('equal')
    plt.tight_layout()

    plt.savefig(greendir+'/check_trace_'+trace_name+'.png', dpi=300)

