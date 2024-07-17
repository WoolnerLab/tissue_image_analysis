"""
handtrace.py

code for running traces from hand

Created by Natasha Cowley on 2023-01-24, adapted from code by Emma Johns and Alex Nestor-Bergmann
"""
from glob import glob
import pandas as pd
import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
import itertools
from skan.csr import skeleton_to_csgraph, make_degree_image, pixel_graph
from skan import Skeleton, summarize, draw
from skan.pre import threshold
import skimage.morphology as sk
from skimage.filters import rank
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import os

from source import segmentation_hand

def run_trace(edges_name, input_dir):
    
    edges_file=input_dir+edges_name+'.tif'
    image0 = iio.v2.imread(edges_file)
    #skeletonise and segment edges
    im, data=segmentation_hand.skeletonise(edges_file, cp=False)
    #remove spiderlegs/contractible branches
    edge_verts, n_coords=segmentation_hand.extract_edges_verts(data)

    Nv=len(n_coords)
    Ne=len(edge_verts)
    Nc=2+Ne-Nv-1 #Euler characteristic for a planar graph minus the outside infinite face.
    print(Nv, Ne, Nc)

    R=np.transpose(np.vstack((n_coords[:,1],n_coords[:,0])))


    A=segmentation_hand.construct_ev_incidence_matrix(edge_verts, Ne, Nv)
    G=segmentation_hand.construct_adjacency_matrix(A)
    cells=segmentation_hand.get_cycles(G, 12)

    if len(cells)!=Nc: print("Nc-len(cells) = ", Nc-len(cells))
        
    cell_edges=segmentation_hand.assign_edges_to_cells(cells, edge_verts, Ne)


    B=segmentation_hand.construct_ce_incidence_matrix(cells,edge_verts,cell_edges, Nc, Ne, R)

    if len(np.where(B@A!=0)[0])!=0: print("Matrix generation error, check A and B")
        
    C=0.5*(abs(B)@abs(A))



    return edge_verts,cells, cell_edges, A, B, C, R, image0




def check_trace_plot(savedir,image0, cells, R, edges_name):
    """
    Plots the network configuration to allow for visual checking.
    """

    plt.figure(figsize=(15,15))
    fig, ax = plt.subplots()
    ax.imshow(image0, cmap='gray');
    for i in range(len(cells)):
        ax.plot(R[np.append(cells[i], cells[i][0])][:,0],R[np.append(cells[i], cells[i][0])][:,1])
    plt.savefig(savedir+'/segmentation_cells_'+edges_name+'.png', bbox_inches="tight", dpi=300)
    plt.close()