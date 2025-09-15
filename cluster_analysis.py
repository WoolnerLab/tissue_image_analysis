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
from src import geometry
from src import mechanics
from src import fileio
from src import visualise

CURRENT_DIR = os.getcwd()
input_dir=CURRENT_DIR+'/Input/'

if os.path.exists(CURRENT_DIR+'/Output/')==False: os.mkdir(CURRENT_DIR+'/Output/')
output_dir=CURRENT_DIR+'/Output/'

#########################
#User Input
#########################
trace_name='20250805_1_YJ_BFPCAAX-mCheHis-A4-KrasD12_MP_trace.tif'
trace_file=sorted(glob(input_dir+trace_name))[0] #trace filename
boundary_file=glob(input_dir+'*_boundary.tif')[0] #boundary filename
exp_id=trace_name.split('_')[0]+'_'+trace_name.split('_')[1]+'_'+trace_name.split('_')[2]
print(exp_id)

pixel_size=512
micron_size=739.54
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

# shell_matrix=geometry.get_nn_shells(B) #Comment out if not needed (quite slow)
# np.savetxt(matrix_dir+'/'+exp_id+ "_fr%03d"%frame +"nn_shells.txt",shell_matrix)

fileio.write_matrices(matrix_dir,A, B, C,R, exp_id, 1)
fileio.write_cell_data(trace_dir,edge_verts, cell_edges, cells, exp_id, 1)

#################################
# Process boundary file
#################################
b_image = iio.v2.imread(boundary_file)

#skeletonise boundary
b_skel = sk.skeletonize(b_image, method='lee')

#get pixel coordinates of boundary
b_pixel_graph, b_coordinates = skeleton_to_csgraph(b_skel)
boundary_data = summarize(Skeleton(b_skel))

#########################
# Find shortest distance from cell centres to edges
#########################

cell_edge_count=geometry.get_edge_count(B)
cell_centres_pixel=geometry.get_cell_centres(C,R,cell_edge_count)


b_coords=np.transpose(b_coordinates)[:,[1,0]]
dists=cdist(cell_centres_pixel, b_coords) #distances between cell centres and boundary points
pt_id=np.array([np.where(n==np.min(n))[0][0] for n in  dists]) #get's index of nearest boundary point



line_to_cluster=cell_centres_pixel-b_coords[pt_id]

line_angle=geometry.get_line_cluster_angle(line_to_cluster)

cluster_angle=line_angle-np.pi/2 #cluster angle is perpendicular to line angle
cluster_angle=np.where(cluster_angle<0, cluster_angle+np.pi, cluster_angle) #get cluster angle in 0-pi range.


#########################
# Get geometric data
#########################

#make sure pixel size is size of tif not of trace as some have borders
R=R*(micron_size/pixel_size)
N_c=C.shape[0]
cell_id=np.array(range(N_c))
cell_areas=geometry.get_areas(cells, R)
cell_perimeters=geometry.get_perimeters(A,B,R)
cell_centres=geometry.get_cell_centres(C,R,cell_edge_count)
mean_cell_area=geometry.get_mean_area(cell_areas)      
print("Mean area = ", mean_cell_area)
shape_tensor= geometry.get_shape_tensors(R, cells,cell_edge_count, cell_centres)
circularity, evals=geometry.get_circularity(shape_tensor)
long_axis_angle=geometry.get_shape_axis_angle(shape_tensor)
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

angle_with_boundary=long_axis_angle-cluster_angle

alignment=np.cos(angle_with_boundary)**2

awb_0_pi=np.where(angle_with_boundary<0, angle_with_boundary+np.pi, angle_with_boundary)
awb_0_pi2=np.where(awb_0_pi>np.pi/2,-1*(awb_0_pi-np.pi), awb_0_pi)

#########################
# Data output
#########################

##Cell Data

geom_data_names=['cell_id', 'cell_area_microns', 'cell_perimeter_microns', 'cell_centre_x', 'cell_centre_y','shape_parameter', 'circularity', 'cell_edge_count', \
'long_axis_angle', 'angle_with_boundary','alignment']
cell_data_geom=np.transpose(np.vstack((cell_id, cell_areas, cell_perimeters,cell_centres[:,0], cell_centres[:,1],\
shape_parameter, circularity, cell_edge_count, long_axis_angle, awb_0_pi2, alignment)))

cell_df=pd.DataFrame(cell_data_geom, columns=geom_data_names)

#write geometry data frames and summary stats
cell_df.to_csv(data_dir + '/'+exp_id +'_cell_data_cluster_angle.csv', index=False)
cell_df.iloc[:,1:].describe().to_csv(data_dir + '/'+exp_id+'cluster_angle_summary_stats.csv')




#########################
# Plots
#########################

plot_variable='alignment'
plot_label='Alignment'
cmap='viridis'

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
polys=visualise.plot_polys(C, R, cell_centres)
polys.set_array(cell_df[plot_variable])
polys.set_cmap(cmap) ###set polygon colourmap here
polys.set_edgecolor('black') #black edges
polys.set_clim(np.min(cell_df[plot_variable]),np.max(cell_df[plot_variable]))
ax.add_collection(polys) 
cbar = fig.colorbar(polys, ax=ax)
cbar.ax.set_ylabel(plot_label, rotation=90) ###set colorbar label
ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
ax.set_xlim(min(R[:,0]-10), max(R[:,0]+10)) #if no edditional elements polygons won't show without manual setting of axes
ax.set_ylim(min(R[:,1]-10), max(R[:,1]+10))
visualise.plot_alignment_axis(cell_centres,np.asarray(cell_df['long_axis_angle']))

ax.scatter(b_coords[:,0]*(micron_size/pixel_size), b_coords[:,1]*(micron_size/pixel_size), c='k', s=5)

for i in range(N_c):
    ax.plot([b_coords[pt_id[i],0]*(micron_size/pixel_size), b_coords[pt_id[i],0]*(micron_size/pixel_size)+line_to_cluster[i,0]*(micron_size/pixel_size)],[b_coords[pt_id[i],1]*(micron_size/pixel_size), b_coords[pt_id[i],1]*(micron_size/pixel_size)+line_to_cluster[i, 1]*(micron_size/pixel_size)])

plt.tight_layout()

plt.savefig(plot_dir+'/'+exp_id+'_'+plot_variable+'.png', dpi=300) ##edit filename here

