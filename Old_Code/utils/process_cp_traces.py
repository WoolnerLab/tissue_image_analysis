from glob import glob
import pandas as pd
import imageio as iio
import matplotlib.pyplot as plt
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

folder='C:\\Users\\v35431nc\\Documents\\Lab_Stuff\\Movies_to_track\\unstretched\\20231019_1_IP_GFPCAAX-CheHis_uu_0p5_SumP\\5min_int'
files = glob(folder+'\\*_trace.tif')
#cp_cyto=np.load(files[0], allow_pickle=True)
#trace=find_boundaries(cp_cyto.item()['masks'])

for f in files:
    image0_cp =io.imread(f, as_gray=True)
    #skeletonise and segment edges
    trace_name=f.split('\\')[-1].split('_seg')[0]
    im_cp, data_cp=handtrace.skeletonise(f)
    draw.overlay_euclidean_skeleton_2d(im_cp,data_cp,
                                    skeleton_color_source='branch-type', skeleton_colormap='Set2');
    plt.show()
    edge_verts_cp, n_coords_cp=handtrace.extract_edges_verts(data_cp)
    Nv_cp=len(n_coords_cp)
    Ne_cp=len(edge_verts_cp)
    A_cp=handtrace.construct_ev_incidence_matrix(edge_verts_cp, Ne_cp, Nv_cp)
    G_cp=handtrace.construct_adjacency_matrix(A_cp)
    cells_cp=handtrace.get_cycles(G_cp, 12)
    cell_edges_cp=handtrace.assign_edges_to_cells(cells_cp, edge_verts_cp, Ne_cp)

    #For cellpose traces we need to remove the boundary cells
    edge_cells_cp=handtrace.get_edge_cells(cells_cp, edge_verts_cp,Ne_cp)
    int_cells_cp,int_cell_edges_cp, int_edge_verts_cp, int_n_coords_cp=handtrace.get_interior_cells(cells_cp, edge_cells_cp, cell_edges_cp, edge_verts_cp, n_coords_cp)

    #check connectivity

    Nc_cp=len(int_cells_cp)
    print("Nc = ", Nc_cp)
    Ne_cp=len(int_edge_verts_cp)
    Nv_cp=len(int_n_coords_cp)
    print(2+Ne_cp-Nv_cp-1)

    #get matrices

    A_cp=handtrace.construct_ev_incidence_matrix(int_edge_verts_cp, Ne_cp, Nv_cp)
    B_cp=handtrace.construct_ce_incidence_matrix(int_cells_cp,int_edge_verts_cp,int_cell_edges_cp, Nc_cp, Ne_cp)

    if len(np.where(B_cp@A_cp!=0)[0])!=0: print("Matrix generation error, check A and B")
        
    C_cp=0.5*(abs(B_cp)@abs(A_cp))

    R_cp=np.transpose(np.vstack((int_n_coords_cp[:,1],int_n_coords_cp[:,0])))

    #plot extracted cells
    fig, ax = plt.subplots()
    ax.imshow(im_cp, cmap='gray');
    for i in range(len(int_cells_cp)):
        ax.plot(int_n_coords_cp[np.append(int_cells_cp[i], int_cells_cp[i][0])][:,1],int_n_coords_cp[np.append(int_cells_cp[i], int_cells_cp[i][0])][:,0])
    plt.savefig(folder+'/segmentation_cells_'+trace_name+'.png', bbox_inches="tight", dpi=300)
    plt.show()

    #Save matrices
    savedir=folder+'/Matrices'
    if os.path.exists(savedir)==False: os.mkdir(savedir)

    np.savetxt(savedir+"/Matrix_A_"+trace_name+".txt",A_cp)
    np.savetxt(savedir+"/Matrix_B_"+trace_name+".txt",B_cp)
    np.savetxt(savedir+"/Matrix_R_"+trace_name+".txt",R_cp)
    np.savetxt(savedir+"/Matrix_C_"+trace_name+".txt",C_cp)
