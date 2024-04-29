"""
cellmatch.py

Functions to reindex cells when tracking cells over time.

"""
import numpy as np
import pandas as pd

from glob import glob

from utils import fileio
from utils import geometry
from utils import matrices


def reindex_data_tm(o_folder,exp_id, save_dir,stretch, frame, pixel_size, micron_size, cell_id):
    mat_dir=glob(o_folder+'\\'+exp_id+'_'+ stretch + '*_*P_fr*'+frame+'\\*\\Matrices')[0]
    tr_dir=glob(o_folder+'\\'+exp_id+'_'+ stretch + '*_*P_fr*'+frame+'\\*\\Trace_extraction')[0]
    A  = np.loadtxt(glob(mat_dir+'/*Matrix_A*.txt')[0]) # Incidence matrix. Rows => edges; columns => vertices.
    B  = np.loadtxt(glob(mat_dir+'/*Matrix_B*.txt')[0]) # Incidence matrix. Rows => cells; columns => edges.
    C  = np.loadtxt(glob(mat_dir+'/*Matrix_C*.txt')[0]) # Incidence matrix. Rows => cells; columns => vertices. 
    R  = np.loadtxt(glob(mat_dir+'/*Matrix_R*.txt')[0]) # Coordinates of vertices

    edge_verts  = np.loadtxt(glob(tr_dir+'/*edge_verts.csv')[0]).astype(int)
    c_edges = pd.read_csv(glob(tr_dir+'/*cell_edges.csv')[0],header =None, delimiter=',', names=list(range(13))).dropna(axis='columns', how='all')
    c_verts= pd.read_csv(glob(tr_dir+'/*cell_vertices.csv')[0],header =None, delimiter=',', names=list(range(13))).dropna(axis='columns', how='all')
    cell_edges={}
    for i in range(len(c_edges)):
        cell_edges[i]=np.asarray(c_edges.iloc[i].dropna())

    
    all_cells={}
    for i in range(len(c_verts)):
        all_cells[i]=np.asarray(c_verts.iloc[i].dropna())

    #save_dir='C:\\Users\\v35431nc\\Documents\\Lab_Stuff\\Relaxation\\20230203_incremental'

    A_new,B_new,C_new,R_new,cc, new_cells, new_cell_edges=new_matrices(A,B,C,R, edge_verts, cell_edges, all_cells,cell_id)
    fileio.write_matrices(save_dir+'/Matrices',A_new, B_new, C_new,R_new, exp_id+"_" +stretch+"_fr"+frame+"_track")

    R_new=R_new*(micron_size/pixel_size)
    R_new=R_new-np.mean(R_new, axis=0)
    ce_new=geometry.get_edge_count(B_new)
    cc_new=geometry.get_cell_centres(C_new,R_new,ce_new)
    

    return A_new, B_new, C_new, R_new, cc_new, new_cells, new_cell_edges

def new_matrices(A,B,C,R, edge_verts, cell_edges, all_cells,cell_id):
    ce=[cell_edges[x] for x in cell_id.astype(int)]
    cells=[all_cells[x] for x in cell_id.astype(int)]

    cell_edge_count=geometry.get_edge_count(B)
    cell_centres=geometry.get_cell_centres(C,R,cell_edge_count)
    cc=np.array([cell_centres[x] for x in cell_id.astype(int)])

    vertices=np.unique([x for sublist in cells for x in sublist]).astype(int)
    edges=np.unique([x for sublist in ce for x in sublist]).astype(int)

    edge_verts=edge_verts[sorted(edges)]
    R_new=R[sorted(vertices)]

    Nv=len(vertices)
    Ne=len(edges)
    #re index  and edges
    nodes=sorted(vertices)
    new_nodes=np.linspace(0,len(vertices)-1, len(vertices)).astype(int)
    node_map = {nodes[i]: new_nodes[i] for i in range(len(new_nodes))}

    edges=sorted(edges)
    new_edges=np.linspace(0,len(edges)-1, len(edges)).astype(int)
    edge_map = {edges[i]: new_edges[i] for i in range(len(new_edges))}


    edge_verts[:,0]=[node_map[x] for x in edge_verts[:,0]]
    edge_verts[:,1]=[node_map[x] for x in edge_verts[:,1]]

    new_cells=[[node_map[x] for x in sublist] for sublist in cells]
    new_cell_edges=[[edge_map[x] for x in sublist] for sublist in ce]

    new_A,new_B,new_C,new_R=matrices.get_matrices(edge_verts, R_new, new_cell_edges, cc[:,0], cc[:,1])

   

    return new_A, new_B, new_C, new_R, cc, new_cells, new_cell_edges