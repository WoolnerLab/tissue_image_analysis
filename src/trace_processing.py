"""
trace_processing.py
Natasha Cowley 2024/07/16

Functions to read in traces and manipulate data into matrices

"""

from glob import glob
import pandas as pd
import numpy as np
import itertools
from skan import Skeleton, summarize
from skan.csr import skeleton_to_csgraph
import skimage.morphology as sk
import networkx as nx
from skimage import io
from scipy.spatial.distance import cdist
import shapely 

def skeletonise(filename, smooth=True):
    """
    Function skeletonises trace, filtering is optimised for 1024 images, 
    different processing may work better for images of lower resolution.
    """

    # Read the line.
    image = io.imread(filename, as_gray=True)
    # Set all active pixels to have value 1.

    image[image < np.mean(image[image > 0])/4] = 0
    image[image > 0] = 1

    if smooth:
 
        image = sk.binary_dilation(image)

    image = sk.binary_closing(image)

    if smooth:
        image = sk.skeletonize(image, method='lee')
    if not smooth:
        image = sk.medial_axis(image, return_distance=False)
    
    branch_data = summarize(Skeleton(image))
    return image, branch_data

def flipper_mask(skel, branch_data, lifetime):
    pixel_graph, coordinates=skeleton_to_csgraph(skel) #get coordinates from skeleton
    coords=np.transpose(coordinates)

    non_nan_pixels=np.transpose(np.where(~np.isnan(lifetime))) #find pixels in masked lifetime image

    dist_matrix=cdist(non_nan_pixels, coords) #Distance between lifetime pixels and skeleton coordinates
    min_dists_ind=np.argmin(dist_matrix, axis=1)
    min_dists=np.min(dist_matrix, axis=1)
    #Make edge-node matrix
    skeleton=Skeleton(skel)
    p_list=skeleton.paths_list()
    path_node_matrix=np.zeros((len(p_list), len(coords)))
    for n in range(len(p_list)):
        path_node_matrix[n, p_list[n]]=1
    #find nearest edge to each pixel
    nearest_path=np.array([np.argwhere(path_node_matrix[:,min_dists_ind[n]]!=0)[0][0] for n in range(len(min_dists))])
    nearest_path=np.where(min_dists<3, nearest_path, np.nan)
    #find mean lifetime along each edge
    pts_per_edge=np.array([len(np.ravel(lifetime[~np.isnan(lifetime)])[nearest_path==n]) for n in range(len(p_list))])
    mean_lt=np.array([np.nanmean(np.ravel(lifetime[~np.isnan(lifetime)])[nearest_path==n]) for n in range(len(p_list))])
    median_lt=np.array([np.nanmedian(np.ravel(lifetime[~np.isnan(lifetime)])[nearest_path==n]) for n in range(len(p_list))])
    point_density=pts_per_edge/branch_data['branch-distance'].values

    mean_lt=np.where(pts_per_edge>4,mean_lt , np.nan)
    mean_lt=np.where(point_density>0.5,mean_lt , np.nan)
    median_lt=np.where(pts_per_edge>4,median_lt , np.nan)
    median_lt=np.where(point_density>0.5,median_lt , np.nan)



    branch_data['mean_lt']=mean_lt
    branch_data['median_lt']=median_lt


    data_internal=branch_data[branch_data['branch-type']==2] #do we also need to split out sub skeletons for disconnected traces? We also want to identify holes by number of cycles compared to Nc, if cycles < Nc then there is a hole and can't get matrices.
    data_int=data_internal.drop_duplicates().reset_index(drop=True)

    return data_int.mean_lt, data_int.median_lt

def extract_edges_verts(data):
    data_internal=data[data['branch-type']==2] #do we also need to split out sub skeletons for disconnected traces? We also want to identify holes by number of cycles compared to Nc, if cycles < Nc then there is a hole and can't get matrices.
    data_int=data_internal.drop_duplicates().reset_index(drop=True)
    
    #find unique vertices
    src=data_int[['node-id-src', 'image-coord-src-1', 'image-coord-src-0']].drop_duplicates()
    dst=data_int[['node-id-dst', 'image-coord-dst-1', 'image-coord-dst-0']].drop_duplicates()

    src=src.rename(columns={'node-id-src':'node_id', 'image-coord-src-1':'x', 'image-coord-src-0':'y'})
    dst=dst.rename(columns={'node-id-dst':'node_id', 'image-coord-dst-1':'x', 'image-coord-dst-0':'y'})

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

    edge_verts=np.sort(edge_verts, axis=1) #orientate edges (for assigning values in the A incidence matrix later)
    
    return edge_verts, n_coords

def construct_ev_incidence_matrix(edge_verts, Ne, Nv):
    #Find the edge-vetex incidence matrix A_jk

    A=np.zeros((Ne,Nv))

    #arbitrarily set orientation by order of verts in list.
    for j in range(0,Ne):
        A[j][edge_verts[j, 0]]=-1 #flows out of vertex
        A[j][edge_verts[j, 1]]=1 #flows into vertex
        
    return A


def get_cycles(G, n_threshold, R):
       #For small graphs we can use the minimum_cycle_basis function below, but it is very very slow for most images:
    #cells=nx.minimum_cycle_basis(G)

    cycles=list(nx.simple_cycles(G, n_threshold))
    
    #remove duplicates (cells share 1 edge/2 vertices)
  
    c=sorted(list(cycles for cycles,_ in itertools.groupby(cycles)), key=len) 
    cells=[c[0]]
    for i in c:
         if all([len(set(i).intersection(x))<3 for x in cells]):
                cells.append(i)
                
    #Find make cell orientation uniform            
    cell_orientation=[shapely.Polygon(R[cells[x]]).exterior.is_ccw for x in range(len(cells))]
    cells=[cells[x][::-1] if cell_orientation[x] else cells[x] for x in range(len(cells))]

    
    return cells



def construct_ce_incidence_matrix(cells,edge_verts,A): 
    Nc=len(cells)
    Ne=len(edge_verts)
    B=np.zeros((Nc,Ne))

    for i in range(0,Nc): #loop over cells
    
        for j in range(len(cells[i])): #loop over each cell verts
            
            #get each edge in a cell in order
            edge=list(set(np.argwhere(A[:,cells[i][j-1]]).flatten()).intersection(np.argwhere(A[:,cells[i][j]]).flatten()))[0]
            #cells is a list of vertices in cells in order round cell\
            #if the 2nd vert in an edge from edge_verts(used to make A) == next vert in cells\
            #then orientation matches, B[i, edge]=1, otherwise orientation is oposite B[i, egde]=-1
            if edge_verts[edge,1]==cells[i][j]: 
                B[i, edge]=1
            else:
                B[i, edge]=-1

    return B

def get_matrices(trace_file, flipper=False, flipper_file=""):
    
    #skeletonise and segment edges
    im, data=skeletonise(trace_file)
    if flipper==True:
 
        lifetime=io.imread(flipper_file, as_gray=True)
        mean_lt, median_lt=flipper_mask(im, data, lifetime)
    
    #remove spiderlegs/contractible branches
    edge_verts, n_coords=extract_edges_verts(data)

    Nv=len(n_coords)
    Ne=len(edge_verts)
    Nc=2+Ne-Nv-1 #Euler characteristic for a planar graph minus the outside infinite face. (only works for connected graphs)
    #print(Nv, Ne, Nc)

    R=np.transpose(np.vstack((n_coords[:,0],n_coords[:,1])))


    A=construct_ev_incidence_matrix(edge_verts, Ne, Nv)
        
    #construct graph from edgelist
    G=nx.Graph([tuple(edge_verts[x]) for x in range(len(edge_verts))])

    cells=get_cycles(G, 12, R)

    #if len(cells)!=Nc: print("Nc-len(cells) = ", Nc-len(cells))

    if set([x for xs in cells for x in xs]).difference(set(range(Nv)))!=set(): print("hanging vertex, check spider legs")

    B=construct_ce_incidence_matrix(cells,edge_verts, A)

    if len(np.where(B@A!=0)[0])!=0: print("Matrix generation error, check A and B")
        
    C=0.5*(abs(B)@abs(A))

    cell_edges=[np.where(B[x, :]!=0) for x in range(len(B))]

    if flipper==False:
        return R, A, B, C, G, cells, edge_verts, cell_edges
    else:
        return R, A, B, C, G, cells, edge_verts, cell_edges, mean_lt, median_lt

