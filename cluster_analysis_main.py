"""
main.py
Natasha Cowley 2024/07/16


"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

import imageio as iio
from skan.csr import skeleton_to_csgraph, make_degree_image, pixel_graph
from skan import Skeleton, summarize, draw
import skimage.morphology as sk
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

from glob import glob
from datetime import datetime


from src import trace_processing
from src import cluster_analysis
from src import geometry
from src import mechanics
from src import fileio
from src import visualise

CURRENT_DIR = os.getcwd()
input_dir=CURRENT_DIR+'/Input/'
input_dir="/home/ncowley/Documents/Work/Kras/1-3cells away/20250828_KrasV12/"

if os.path.exists(CURRENT_DIR+'/Output/')==False: os.mkdir(CURRENT_DIR+'/Output/')
output_dir=CURRENT_DIR+'/Output/'

#########################
#User Input
#########################

###Put trace file and boundary file in input folder

trace_name='20250828_1_YJ_BFPCAAX-mCheHis-A4-KrasV12_MP_seg_cp_trace.tif'  #Important that 1st part is date, 2nd part is experiment number that day, 3rd part is initials
exp_id=trace_name.split('_')[0]+'_'+trace_name.split('_')[1]+'_'+trace_name.split('_')[2]
trace_file=sorted(glob(input_dir+trace_name))[0] #trace filename
boundary_file=glob(input_dir+exp_id+'*_boundary.tif')[0] #boundary filename (make sure it ends 'boundary' )
cluster_file=glob(input_dir+exp_id+'*_cluster.tif')[0] #cluster filename (make sure it ends 'cluster' )
print(exp_id)

#enter pixel and micron dimensions of original image
pixel_size=512
micron_size=775.0
####################################
# Read files and set up directories
####################################

#make directories to output t
if os.path.exists(output_dir+exp_id)==False: os.mkdir(output_dir+exp_id)

mydir = os.path.join(output_dir,exp_id, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.mkdir(mydir)

trace_dir=mydir+"/Trace_extraction"
matrix_dir=mydir+"/Matrices"
data_dir=mydir+"/Data"
plot_dir=mydir+"/Plots"

if os.path.exists(trace_dir)==False: os.mkdir(trace_dir)
if os.path.exists(matrix_dir)==False: os.mkdir(matrix_dir)
if os.path.exists(data_dir)==False: os.mkdir(data_dir)
if os.path.exists(plot_dir)==False: os.mkdir(plot_dir)

#################################
# Process trace and get matrices
#################################
R, A, B, C, G, cells, edge_verts, cell_edges=trace_processing.get_matrices(trace_file)

cell_edge_count=geometry.get_edge_count(B)
cell_centres=geometry.get_cell_centres(C,R,cell_edge_count)

cluster_cells=cluster_analysis.identify_cluster_cells(cluster_file, cell_centres)

#remove external cells and map cell ids
R, A, B, C, cells, edge_verts, cell_edges, cluster_cells=cluster_analysis.process_traced_cells(R,A,B,C,cells,edge_verts, cell_edges, cluster_cells)

fileio.write_matrices(matrix_dir,A, B, C,R, exp_id, 1)
fileio.write_cell_data(trace_dir,edge_verts, cell_edges, cells, exp_id, 1)

shell_matrix=geometry.get_nn_shells(B)

cell_edge_count=geometry.get_edge_count(B)
cell_centres=geometry.get_cell_centres(C,R,cell_edge_count)
cells_outside_cluster, boundary_edges=cluster_analysis.get_non_cluster_cells(R,B, edge_verts, cluster_cells, cell_edge_count, cell_centres)

boundary_shells=cluster_analysis.get_boundary_shells(B,shell_matrix,cluster_cells, cells_outside_cluster,boundary_edges)

N_c=C.shape[0]
cell_id=np.array(range(N_c))
wild_cells=np.array(list(set(cell_id).difference(cluster_cells)))


#################################
# Process boundary file
#################################
#extract boundary pixels from image and order them in order to fit a spline.
ordered_paths, ordered_cycles=cluster_analysis.process_boundary(boundary_file)
path_pts, cycle_pts, all_pts, all_der=cluster_analysis.fit_spline(ordered_paths, ordered_cycles)

#########################
# Find shortest distance from cell centres to boundary 
#########################
dists=cdist(cell_centres, all_pts) #distances between cell centres and boundary points
min_dists=np.array([np.min(n) for n in  dists])
pt_id=np.array([np.where(n==np.min(n))[0][0] for n in  dists]) #get's index of nearest boundary point
line_to_cluster=cell_centres-all_pts[pt_id]
boundary_tangent=all_der[pt_id]/np.linalg.norm(all_der[pt_id], axis=1)[...,np.newaxis]
boundary_angle=np.arctan2(boundary_tangent[:,1], boundary_tangent[:,0])
boundary_angle=np.where(boundary_angle<0, boundary_angle+np.pi, boundary_angle)

#########################
# Get geometric data
#########################

#make sure pixel size is size of tif not of trace as some have borders
R=R*(micron_size/pixel_size)
cell_centres=geometry.get_cell_centres(C,R,cell_edge_count)
cell_areas=geometry.get_areas(cells, R)
cell_perimeters=geometry.get_perimeters(A,B,R)
mean_cell_area=geometry.get_mean_area(cell_areas)      
print("Mean area = ", mean_cell_area)
shape_tensor= geometry.get_shape_tensors(R, cells,cell_edge_count, cell_centres)
circularity, evals=geometry.get_circularity(shape_tensor)
long_axis_angle, long_axis=geometry.get_shape_axis_angle(shape_tensor)
shape_parameter = cell_perimeters/(np.sqrt(cell_areas))
tangents=geometry.get_tangents(A,R)
edge_lengths=geometry.get_edge_lengths(tangents)
Q,J=geometry.get_Q_J(tangents, edge_lengths, B,cell_perimeters)
zeta=np.sqrt(-np.linalg.det(J))
eval_minor= evals[:,0]
eval_major= evals[:,1]
edge_angles=geometry.get_edge_angle(tangents)


#########################
# Get difference in angles
#########################

angle_with_boundary=np.arccos(np.sum(long_axis*boundary_tangent,axis=1))
angle_with_boundary=np.where(angle_with_boundary>np.pi/2, np.pi-angle_with_boundary, angle_with_boundary)
#########################
# Data output
#########################

cell_type=np.zeros(N_c).astype(str)
cell_type[cluster_cells]="cluster"
cell_type[wild_cells]="wild"


##Cell Data geometric quantities for all cells

all_data_names=['cell_id','cell_type', 'cell_area_microns', 'cell_perimeter_microns', 'cell_centre_x', 'cell_centre_y','shape_parameter', 'circularity', 'cell_edge_count', \
'long_axis_angle']
cell_data_all=np.transpose(np.vstack((cell_id, cell_type, cell_areas, cell_perimeters,cell_centres[:,0], cell_centres[:,1],\
shape_parameter, circularity, cell_edge_count, long_axis_angle)))

cell_df=pd.DataFrame(cell_data_all, columns=all_data_names)

#write geometry data frames and summary stats
cell_df.to_csv(data_dir + '/'+exp_id +'_cell_data_geometry.csv', index=False)


##Cells outside cluster
outside_data_names=['cell_id', 'cell_area_microns', 'cell_perimeter_microns', 'cell_centre_x', 'cell_centre_y','shape_parameter', 'circularity', 'cell_edge_count', \
'long_axis_angle', 'angle_with_boundary', 'boundary_shell', 'distance_to_boundary']
cell_data_outside=np.transpose(np.vstack((cell_id[cells_outside_cluster], cell_areas[cells_outside_cluster], cell_perimeters[cells_outside_cluster],cell_centres[cells_outside_cluster,0], cell_centres[cells_outside_cluster,1],
shape_parameter[cells_outside_cluster], circularity[cells_outside_cluster], cell_edge_count[cells_outside_cluster], long_axis_angle[cells_outside_cluster], angle_with_boundary[cells_outside_cluster],boundary_shells, min_dists[cells_outside_cluster]*(micron_size/pixel_size) )))

outside_df=pd.DataFrame(cell_data_outside, columns=outside_data_names)
outside_df.to_csv(data_dir + '/'+exp_id +'_cells_outside_cluster_orientation.csv', index=False)




#########################
# Plots
#########################


plot_cells=cells_outside_cluster[np.where(boundary_shells<3)[0]] #shells 1 and 2
plot_label='Angle with boundary'
cmap='Blues'

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
polys=visualise.plot_polys(C[plot_cells], R, cell_centres[plot_cells])
polys.set_array(angle_with_boundary[plot_cells])
polys.set_cmap(cmap) ###set polygon colourmap here
polys.set_edgecolor('black') #black edges
polys.set_clim(0,np.pi/2)
ax.add_collection(polys) 
cbar = fig.colorbar(polys, ax=ax)
cbar.ax.set_ylabel(plot_label, rotation=90) ###set colorbar label
ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
ax.set_xlim(min(R[:,0]-10), max(R[:,0]+10)) #if no edditional elements polygons won't show without manual setting of axes
ax.set_ylim(min(R[:,1]-10), max(R[:,1]+10))
visualise.plot_alignment_axis(cell_centres[plot_cells],long_axis_angle[plot_cells], c='k')
visualise.plot_edges(A[boundary_edges],R, c='k')
for i in range(len(path_pts)):
    plt.plot(path_pts[i][0]*(micron_size/pixel_size), path_pts[i][1]*(micron_size/pixel_size), c='r')

for i in range(len(cycle_pts)):
    plt.plot(cycle_pts[i][0]*(micron_size/pixel_size), cycle_pts[i][1]*(micron_size/pixel_size), c='r')

plt.title("shells 1-2")
plt.tight_layout()

plt.savefig(plot_dir+'/'+exp_id+'_shells_1-2.png', dpi=300) ##edit filename here

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

polys_low,polys_high=visualise.plot_binary_polys(C, R, cell_centres, np.where(cell_type=='cluster',1,0), 1)
polys_low.set_facecolor('red') 
polys_low.set_alpha(0.25)
polys_low.set_edgecolor('black') 
ax.add_collection(polys_low)

polys_high.set_facecolor('green') 
polys_high.set_alpha(0.25)
polys_high.set_edgecolor('black') 
ax.add_collection(polys_high)

ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
ax.set_xlim(min(R[:,0]-10), max(R[:,0]+10)) #if no edditional elements polygons won't show without manual setting of axes
ax.set_ylim(min(R[:,1]-10), max(R[:,1]+10))

visualise.plot_cell_id(cell_centres)
ax.set_title("{}".format('cell IDs')) ###change title

plt.tight_layout()

plt.savefig(plot_dir+'/'+exp_id+'_'+'cell_ids'+'.png', dpi=300) ##edit filename here
plt.close()

