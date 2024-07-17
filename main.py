"""
main.py
Natasha Cowley 2024/07/16


"""
import os
import numpy as np

from glob import glob

from src import trace_processing
from src import geometry
from src import mechanics
from src import fileio
from src import visualise

CURRENT_DIR = os.getcwd()
#input_dir=CURRENT_DIR+'/Input/'
input_dir='C:\\Users\\v35431nc\\Documents\\Lab_Stuff\\Parameter_Inference/Traces/'

output_dir=CURRENT_DIR+'/Output/'

#########################
#User Input
#########################

c_file=sorted(glob.glob(input_dir+'*_trace_conf.csv'))[0]
f = open(c_file,'r')
lines = f.read().splitlines()[1:]
f.close()

#########################
#Constant variables
#########################
Lambda = -0.259 # Line tension (tunes P_0) (non-dimensional)
Gamma = 0.172  # Contractility (non-dimensional)
L_0  = -Lambda/(2*Gamma)  # Cell preferred perimeter (non-dimensional)
area_scaling = 0.00001785     # Gradient of the change in prefArea, used for time scaling. NB: if running code for knock downs you may need to recalculate this number.


####################################
# Read files and set up directories
####################################

for l in lines:

    #read in conf file
    edges_name,t_min, pixel_size, micron_size = fileio.read_conf_line(l)
    exp_id=edges_name.split('_')[0]+'_'+edges_name.split('_')[1]+'_'+edges_name.split('_')[7]
    t=t_min*60.0
    stretch_type=edges_name.split('_')[4][-1]

    trace_file=os.path.join(input_dir, edges_name+".tif")

    #make directories to output to
    save_dir, trace_dir, matrix_dir, data_dir, plot_dir = fileio.setup_directories(output_dir, edges_name)

    #################################
    # Process trace and get matrices
    #################################
    R, A, B, C, G, cells, edge_verts=trace_processing.get_matrices(trace_file)
    shell_matrix=geometry.get_nn_shells(B)
    np.savetxt(matrix_dir+'/'+exp_id+"nn_shells.txt",shell_matrix)

    #########################
    # Get geometric data
    #########################

    #make sure pixel size is size of tif not of trace as some have borders
    R=R*(micron_size/pixel_size)

    cell_areas=geometry.get_areas(A,B, R)
    cell_perimeters=geometry.get_perimeters(A,B,R)
    cell_edge_count=geometry.get_edge_count(B)
    cell_centres=geometry.get_cell_centres(C,R,cell_edge_count)
    mean_cell_area=geometry.get_mean_area(cell_areas)      
    print("Mean area = ", mean_cell_area)
    shape_tensor= geometry.get_shape_tensors(R, cells,cell_edge_count, cell_centres)
    circularity=geometry.get_circularity(shape_tensor)
    long_axis_angle=geometry.get_shape_axis_angle(shape_tensor)
    shape_parameter = cell_perimeters/(np.sqrt(cell_areas))
    tangents=geometry.get_tangents(A,R)
    edge_lengths=geometry.get_edge_lengths(tangents)
    Q,J=geometry.get_Q_J(tangents, edge_lengths, B,cell_perimeters)
    zeta=np.sqrt(-np.linalg.det(J))


    #########################
    # Non-dimensionalisation
    #########################

    A_0dim=#which ever metric we decide here

    R_nd=R/(np.sqrt(A_0dim))
    areas_nd=cell_areas/A_0dim
    perimeters_nd=cell_perimeters/(np.sqrt(A_0dim))
    tangents_nd=geometry.get_tangents(A,R_nd)
    edge_lengths_nd=geometry.get_edge_lengths(tangents_nd)
    cell_centres_nd=geometry.get_cell_centres(C,R_nd,cell_edge_count)


    #########################
    # Get mechanical data
    #########################

    cell_pressures = mechanics.get_cell_pressures(areas_nd)
    cell_tensions = mechanics.get_cell_tensions(Gamma, perimeters_nd, L_0)

