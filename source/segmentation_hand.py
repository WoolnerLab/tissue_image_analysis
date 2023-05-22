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

def skeletonise(filename, smooth=True, cp=True):
    # Read the line.
    image = iio.v2.imread(filename)
    # Set all active pixels to have value 1.
   
    image[image < np.mean(image[image > 0])/4] = 0
    image[image > 0] = 1

    if smooth:
 
        #image=sk.binary_dilation(image, footprint=[[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        image=sk.binary_dilation(image, footprint=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        #image = sk.binary_dilation(image)
   

    image = sk.binary_closing(image)


    if smooth:
        image = sk.skeletonize(image)
    if not smooth:
        image = sk.medial_axis(image, return_distance=False)
    

    pixel_graph, coordinates = skeleton_to_csgraph(image)
    branch_data = summarize(Skeleton(image))
    return image, branch_data


def extract_edges_verts(data):
    data_internal=data[data['branch-type']==2] #do we also need to split out sub skeletons for disconnected traces? We also want to identify holes by number of cycles compared to Nc, if cycles < Nc then there is a hole and can't get matrices.
    data_int=data_internal.drop_duplicates().reset_index(drop=True)
    
    #find unique vertices
    src=data_int[['node-id-src', 'image-coord-src-0', 'image-coord-src-1']].drop_duplicates()
    dst=data_int[['node-id-dst', 'image-coord-dst-0', 'image-coord-dst-1']].drop_duplicates()

    src=src.rename(columns={'node-id-src':'node_id', 'image-coord-src-0':'y', 'image-coord-src-1':'x'})
    dst=dst.rename(columns={'node-id-dst':'node_id', 'image-coord-dst-0':'y', 'image-coord-dst-1':'x'})

    node_coords=pd.concat([src,dst]).drop_duplicates().reset_index(drop=True)
    nodes=np.array(sorted(node_coords['node_id']))
   
    #edges as vertex pairs
    edge_verts=np.array(data_int[['node-id-src', 'node-id-dst']])
    
    #re index nodes
    new_nodes=np.linspace(0,len(node_coords)-1, len(node_coords)).astype(int)
    node_map = {nodes[i]: new_nodes[i] for i in range(len(new_nodes))}
    node_coords['node_id']=[node_map[x] for x in node_coords['node_id']]
    n_coords=np.array(node_coords.sort_values(by=['node_id']).iloc[:,1:]) #image coordinates of nodes
    
    #use map to reindex list of verices connected to an edge

    edge_verts[:,0]=[node_map[x] for x in edge_verts[:,0]]
    edge_verts[:,1]=[node_map[x] for x in edge_verts[:,1]]
    
    return edge_verts, n_coords

def construct_ev_incidence_matrix(edge_verts, Ne, Nv):
    #Find the edge-vetex incidence matrix A_jk

    A=np.zeros((Ne,Nv))
    max_v=np.max(edge_verts, axis=1)
    min_v=np.min(edge_verts, axis=1)

    for j in range(0,Ne):
        A[j][max_v[j]]=1 #flows into vertex
        A[j][min_v[j]]=-1 #flows out of vertex
        
    return A

def construct_adjacency_matrix(A):
    Adj=np.dot(np.transpose(abs(A)),abs(A)) #create vertex-vertex adjacency matrix
    Adj[np.where(Adj>1)]=0
    Adj_sp=csr_matrix(Adj)
    G = nx.from_scipy_sparse_array(Adj_sp) #create networkx graph object from sparse adjacency matrix
    
    return G

def get_cycles(G, n_threshold):
       #For small graphs we can use the minimum_cycle_basis function below, but it is very very slow for most images:
    #cells=nx.minimum_cycle_basis(G)
    
    #in order to use simple_cycles function we must have a directed graph
    DG=nx.to_directed(G) #transform to symmetric directed graph
    cycles=[x for x in list(nx.simple_cycles(DG, length_bound=n_threshold)) if len(x)>2 ] #find simple cycles upto length 12 edges and remove links interoduced by directed graph
    #remove duplicates
    cycles.sort()
    c=sorted(list(cycles for cycles,_ in itertools.groupby(cycles)), key=len) 
    cells=[c[0]]
    for i in c:
         if all([len(set(i).intersection(x))<3 for x in cells]):
                cells.append(i)
    
    return cells

def assign_edges_to_cells(cells, edge_verts, Ne):
    cell_edges={}
    for i in range(len(cells)):
        for j in range(Ne):
            if(set(edge_verts[j]).issubset(cells[i])):
                if i in cell_edges.keys():
                    cell_edges[i]=list(set(cell_edges[i]).union([j]))
                else:
                    cell_edges[i]=[j]
    return cell_edges

def get_edge_cells(cells, edge_verts,Ne):
    #get edges connected to each cell, 1 for peripheral edges, 2 for other edges
    edge_cells={}
    for i in range(len(cells)):
        for j in range(Ne):
            if(set(edge_verts[j]).issubset(cells[i])):
                if j in edge_cells.keys():
                    edge_cells[j]=list(set(edge_cells[j]).union([i]))
                else:
                    edge_cells[j]=[i]
    return edge_cells

def construct_ce_incidence_matrix(cells,edge_verts,cell_edges, Nc, Ne):  
    B=np.zeros((Nc,Ne))
    for i in range(Nc):
        for j in range(len(cell_edges[i])):
            ind=np.where(cells[i]==edge_verts[list(cell_edges[i])][j,0])[0][0] #find where first element in edge in cell
            if cells[i][ind - 1]==edge_verts[list(cell_edges[i])][j,1]: #if previous vertex = other edge in cell -> anti clock wise
                B[i, list(cell_edges[i])[j]]=-1
            else:
                B[i,list(cell_edges[i])[j]]=1 #clockwise
    return B

def get_interior_cells(cells, edge_cells, cell_edges, edge_verts, n_coords):
    b_cells=[edge_cells[edge][0] for edge in edge_cells if len(edge_cells[edge])==1] #boundary cells
    l_cells=np.linspace(0, len(cells)-1,len(cells) ).astype(int)
    
    interior_cells=[cells[i] for i in l_cells if i not in set(b_cells)]
    interior_cell_edges=[cell_edges[i]for i in l_cells if i not in set(b_cells)]
    
    int_vertices=np.unique([x for sublist in interior_cells for x in sublist])
    int_edges=np.unique([x for sublist in interior_cell_edges for x in sublist])
    
    int_edge_verts=edge_verts[sorted(int_edges)]
    int_n_coords=n_coords[sorted(int_vertices)]

    #re index  and edges
    nodes=sorted(int_vertices)
    new_nodes=np.linspace(0,len(int_vertices)-1, len(int_vertices)).astype(int)
    node_map = {nodes[i]: new_nodes[i] for i in range(len(new_nodes))}
    
    edges=sorted(int_edges)
    new_edges=np.linspace(0,len(int_edges)-1, len(int_edges)).astype(int)
    edge_map = {edges[i]: new_edges[i] for i in range(len(new_edges))}


    int_edge_verts[:,0]=[node_map[x] for x in int_edge_verts[:,0]]
    int_edge_verts[:,1]=[node_map[x] for x in int_edge_verts[:,1]]

    int_cells=[[node_map[x] for x in sublist]for sublist in interior_cells]
    int_cell_edges=[[edge_map[x] for x in sublist]for sublist in interior_cell_edges]
    
    return int_cells,int_cell_edges, int_edge_verts, int_n_coords
