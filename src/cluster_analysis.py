from glob import glob
import pandas as pd
import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
import itertools
from skan.csr import skeleton_to_csgraph
from skan import Skeleton, summarize
import skimage.morphology as sk
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep,splev,RBFInterpolator
import networkx as nx
from skimage import io, measure
import os
from shapely.geometry import Point, Polygon
import sys
sys.path.insert(0, "..")

from src import geometry


def identify_cluster_cells(cluster_file, cell_centres):
    image_cluster = iio.v2.imread(cluster_file)

    skel_cluster = sk.skeletonize(image_cluster, method='lee')
    pixel_graph, coordinates = skeleton_to_csgraph(skel_cluster)
    cluster_coords=np.transpose(coordinates)[:,[1,0]]
    #identify with segmented cells
    pt_dist=cdist(cluster_coords, cell_centres)
    cluster_cells=np.unique(np.array([np.argmin(n) for n in  pt_dist]))

    return cluster_cells




def process_traced_cells(R,A,B,C,cells,edge_verts, cell_edges, cluster_cells):
    N_c=C.shape[0]
    cell_id=np.array(range(N_c))
    cell_edge_count=geometry.get_edge_count(B)
    cell_centres=geometry.get_cell_centres(C,R,cell_edge_count)

    #remove exterior cells (as from cellpose trace)
    exterior_edges=np.nonzero(np.sum(B, axis=0))[0]
    exterior_cells=np.unique(np.nonzero(B[:, exterior_edges])[0])
    interior_cells=np.array(list(set(cell_id).difference(exterior_cells)))
    interior_edges=np.unique(np.nonzero(B[interior_cells])[1])
    interior_verts=np.unique(np.nonzero(A[interior_edges])[1])

    cluster_cells=np.array(list(set(cluster_cells).intersection(interior_cells)))

    R=R[interior_verts]
    B=B[interior_cells][:,interior_edges]
    A=A[interior_edges][:,interior_verts]
    C=C[interior_cells][:,interior_verts]

    N_c=C.shape[0]
    cell_id=np.array(range(N_c))
    #map new cell, edge and vertex ids
    vert_map= dict(zip(interior_verts, np.linspace(0, len(interior_verts)-1, len(interior_verts), dtype=int)))
    edge_map= dict(zip(interior_edges, np.linspace(0, len(interior_edges)-1, len(interior_edges), dtype=int)))
    cell_map=dict(zip(interior_cells, np.linspace(0, N_c-1, N_c, dtype=int)))
    cells=[[vert_map[v] for v in cells[i]] for i in interior_cells] 
    cell_edges=[[edge_map[e[0]] for e in cell_edges[i]] for i in interior_cells] 
    edge_verts=np.array([[vert_map[v] for v in edge_verts[j]] for j in interior_edges])
    cluster_cells=np.array([cell_map[c] for c in cluster_cells])

    return R, A, B, C, cells, edge_verts, cell_edges, cluster_cells



def get_non_cluster_cells(R,B, edge_verts, cluster_cells, cell_edge_count, cell_centres):
    cell_type=np.zeros(B.shape[0])
    cell_type[cluster_cells]=1
    cell_id=np.array(range(B.shape[0]))
    wild_cells=np.array(list(set(cell_id).difference(cluster_cells)))


    ct_sums=np.array([np.sum(cell_type[np.nonzero(B[:, j])])  for j in range(B.shape[1])]) #sum adjacent cell types over edges

    boundary_edges=np.argwhere(ct_sums==1).ravel() #is edge between cell types or at border
    exterior_edges=np.nonzero(np.sum(B, axis=0))[0] #edges at edge of traceed cells    
    boundary_edges=np.array(list(set(boundary_edges).difference(exterior_edges)))

    #find surounded wild type cells

    B_wild=B[wild_cells]
    Badj_w=abs(B_wild)@abs(B_wild).T-np.diag(cell_edge_count[wild_cells])
    WC_adj=nx.Graph(Badj_w)
    components=[np.array(list(n)) for n in nx.connected_components(WC_adj)]

    component_size=np.array([len(n) for n in nx.connected_components(WC_adj)])

    surrounded=wild_cells[np.hstack([components[n] for n in np.argwhere(component_size<25).ravel()])]

    #get edges of surounded cells
    surrounded_edges=np.unique(np.nonzero(B[surrounded])[1])

    boundary_edges=np.array(list(set(boundary_edges).difference(surrounded_edges)))

    #count surrounded wild type as part of the cluster
    cluster_total=np.array(list(set(cluster_cells).union(surrounded)))
    in_cluster=np.zeros_like(cell_id)
    in_cluster[cluster_total]=1
    wild_out_cells=cell_id[in_cluster==0]
    
    #find convex hull and exclude cells within it

    boundary_edges_all=np.argwhere(ct_sums==1).ravel() 
    boundary_verts_all=np.unique(edge_verts[boundary_edges_all].ravel())
    points_v=R[boundary_verts_all]
    hull_v=ConvexHull(points_v)
    ch=Polygon(points_v[hull_v.vertices])

    wild_out_ch_cells=wild_out_cells[~np.array([ch.contains(Point(i)) for i in cell_centres[wild_out_cells]])]

    return wild_out_ch_cells, boundary_edges



def get_boundary_shells(B,shell_matrix,cluster_cells, wild_out_ch_cells,boundary_edges):
    #find cells in shells from boundary, 1 = next to boundary 

    boundary_cells=np.unique(np.nonzero(abs(B)[:, boundary_edges])[0]) #cells touching the boundary
    boundary_cells_cluster=np.array(list(set(boundary_cells).intersection(cluster_cells)))
    shells=[]
    counted_cells=[]
    for s in range(np.max(shell_matrix[boundary_cells_cluster][:,wild_out_ch_cells]).astype(int)):
        shell_cells=np.unique(np.where(shell_matrix[boundary_cells_cluster][:,wild_out_ch_cells]==s)[1])
        shells.append(np.array(list(set(np.unique(np.where(shell_matrix[boundary_cells_cluster][:,wild_out_ch_cells]==s)[1])).difference(counted_cells))).astype(int))
        counted_cells.extend(shell_cells)


    boundary_shells=np.zeros_like(wild_out_ch_cells)
    for i in range(len(shells)):
        boundary_shells[shells[i]]=i 
    return boundary_shells

def process_boundary(boundary_file):
        #read boundary file
    imageb = iio.v2.imread(boundary_file)
    skel_b = sk.skeletonize(imageb, method='lee')
    b_pixel_graph, b_coordinates = skeleton_to_csgraph(skel_b)
    branch_data = summarize(Skeleton(skel_b), separator='_')
    boundary_coords=np.flip(np.vstack(b_coordinates).T, axis=1)

    #find connected parts of boundary trace and order pixels

    boundary_graph=nx.Graph(b_pixel_graph)

    S_boundary = [boundary_graph.subgraph(c).copy() for c in nx.connected_components(boundary_graph)]

    deg_1_nodes=[[node for node, degree in cc.degree if degree == 1] for cc in S_boundary]

    ordered_components_paths=[]
    ordered_components_cycles=[]

    for cc in range(len(S_boundary)):
        #check for cycles
        cycles=list(nx.cycle_basis(S_boundary[cc]))
        if len(cycles)!=0:
            for c in cycles:
                ordered_components_cycles.append(c)
        #check for endpoints (degree 1 nodes) and find the longest path for all combos
        if len(deg_1_nodes[cc])>1:
            length = dict(nx.all_pairs_dijkstra_path_length(S_boundary[cc]))
            len_nodes=np.array([np.array([length[i][j] for i in deg_1_nodes[cc]]) for j in deg_1_nodes[cc]])            
            source, target=np.array(deg_1_nodes[cc])[np.array(np.unravel_index(np.argmax(len_nodes, axis=None), len_nodes.shape))]
            ordered_components_paths.append(list(nx.all_simple_paths(S_boundary[cc],source, target))[0])
    
    ordered_paths=[boundary_coords[p] for p in ordered_components_paths]
    ordered_cycles=[boundary_coords[c] for c in ordered_components_cycles]


    return ordered_paths, ordered_cycles


def get_cc_contours(edges_file, cell_centres, in_cluster):

    #prepare grid for interpolation
    dims=io.imread(edges_file).shape[0]
    x_edges = np.linspace(0, dims-1, dims)
    y_edges = np.linspace(0, dims-1, dims)
    x_centers = x_edges[:-1] + np.diff(x_edges[:2])[0] / 2.
    y_centers = y_edges[:-1] + np.diff(y_edges[:2])[0] / 2.
    x_i, y_i = np.meshgrid(x_centers, y_centers)
    x_i = x_i.reshape(-1,1)
    y_i = y_i.reshape(-1,1)
    xy_i = np.concatenate([x_i, y_i], axis=1)
    #interpolate cell centre position
    rbf = RBFInterpolator(cell_centres,in_cluster, kernel='linear')
    z_i = rbf(xy_i)
    contours = measure.find_contours(z_i.reshape(dims-1, dims-1), 0.4)
    ordered_paths=[]
    ordered_cycles=[]
    for c in contours:
        if cdist([c[0,:]],[c[-1,:]])[0][0]>0:
            ordered_paths.append(c[:,[1,0]])
        else:
            ordered_cycles.append(c[:,[1,0]])
    
    return ordered_paths, ordered_cycles

def fit_spline(ordered_paths, ordered_cycles, smoothing=25):

    #fit spline to each connected component

    path_pts_s=[]
    path_der_s=[]

    cycle_pts_s=[]
    cycle_der_s=[]

    for p in ordered_paths:

        pts=p
        tck, u = splprep([pts[:,0], pts[:,1]], s=smoothing)

        new_points = splev(u, tck)
        deriv = splev(u, tck, der=1)

        if len(np.nonzero(deriv)[0])!=0:
            path_pts_s.append(new_points)
            path_der_s.append(deriv)


    for c in ordered_cycles:

        pts=c
        tck, u = splprep([pts[:,0], pts[:,1]], s=smoothing, per=1)

        new_points = splev(u, tck)
        deriv = splev(u, tck, der=1)

        if len(np.nonzero(deriv)[0])!=0:
            cycle_pts_s.append(new_points)
            cycle_der_s.append(deriv)


    #combine points into 1 array
    if len(path_pts_s)!=0 and len(cycle_pts_s)!=0:
        path_pts=np.vstack([np.array(p).T for p in path_pts_s])
        path_der=np.vstack([np.array(p).T for p in path_der_s])

        cycle_pts=np.vstack([np.array(c).T[:-1] for c in cycle_pts_s])
        cycle_der=np.vstack([np.array(c).T[:-1] for c in cycle_der_s])

        all_pts=np.vstack((path_pts, cycle_pts))
        all_der=np.vstack((path_der, cycle_der))
    elif(len(path_pts_s)!=0):
        all_pts=np.vstack([np.array(p).T for p in path_pts_s])
        all_der=np.vstack([np.array(p).T for p in path_der_s])
    elif(len(cycle_pts_s)!=0):
        all_pts=np.vstack([np.array(c).T[:-1] for c in cycle_pts_s])
        all_der=np.vstack([np.array(c).T[:-1] for c in cycle_der_s])

    return path_pts_s, cycle_pts_s, all_pts, all_der

