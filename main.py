"""
main.py
Natasha Cowley 2024/07/16


"""
import os
import numpy as np
import pandas as pd
from matplotlib import cm

from glob import glob

from src import trace_processing
from src import geometry
from src import mechanics
from src import fileio
from src import visualise

CURRENT_DIR = os.getcwd()
input_dir=CURRENT_DIR+'/Input/'
#input_dir='C:\\Users\\v35431nc\\Documents\\Lab_Stuff\\Movies_to_track\\100cells/'

output_dir=CURRENT_DIR+'/Output/'
#output_dir='C:\\Users\\v35431nc\\Documents\\Lab_Stuff\\Movies_to_track\\100cells/'

#########################
#User Input
#########################

c_file=sorted(glob(input_dir+'*_conf.csv'))[0]
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
    print(edges_name)
    exp_id=edges_name.split('_')[0]+'_'+edges_name.split('_')[1]+'_'+edges_name.split('_')[2]
    t=t_min*60.0
    stretch_type=edges_name.split('_')[4][-1]
    frame=int(edges_name.split('_')[7][2:])

    trace_file=os.path.join(input_dir, edges_name+".tif")

    #make directories to output to
    save_dir, trace_dir, matrix_dir, data_dir, plot_dir = fileio.setup_directories(output_dir, edges_name)


    #################################
    # Process trace and get matrices
    #################################
    R, A, B, C, G, cells, edge_verts, cell_edges=trace_processing.get_matrices(trace_file)
    shell_matrix=geometry.get_nn_shells(B) #Comment out if not needed (quite slow)
    np.savetxt(matrix_dir+'/'+exp_id+ "_fr%03d"%frame +"nn_shells.txt",shell_matrix)
    fileio.write_matrices(matrix_dir,A, B, C,R, exp_id, frame)
    fileio.write_cell_data(trace_dir,edge_verts, cell_edges, cells, exp_id, frame)
    #########################
    # Get geometric data
    #########################

    #make sure pixel size is size of tif not of trace as some have borders
    R=R*(micron_size/pixel_size)
    N_c=C.shape[0]
    cell_id=np.array(range(N_c))
    cell_areas=geometry.get_areas(cells, R)
    cell_perimeters=geometry.get_perimeters(A,B,R)
    cell_edge_count=geometry.get_edge_count(B)
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
    # Non-dimensionalisation
    #########################

    #We may change how we non-dimensionalise. For now we assume the isotropic stress sums to zero for a monolayer and solve for A_0*
    if t==0 or stretch_type=='u':
        A_0dim= geometry.get_pref_area(cell_areas, Gamma, cell_perimeters, L_0, mean_cell_area) 
        fileio.write_pref_area(data_dir,input_dir, edges_name, A_0dim)
    else:
        A_0dim=fileio.read_pref_area(input_dir, edges_name)
        A_0dim*=(1-area_scaling*t) #scaling stretched cells from Emma's thesis to be revisited

    R_nd=R/(np.sqrt(A_0dim))
    areas_nd=cell_areas/A_0dim
    perimeters_nd=cell_perimeters/(np.sqrt(A_0dim))
    tangents_nd=geometry.get_tangents(A,R_nd)
    edge_lengths_nd=geometry.get_edge_lengths(tangents_nd)
    cell_centres_nd=geometry.get_cell_centres(C,R_nd,cell_edge_count)
  


    #########################
    # Get mechanical data
    #########################

    #must use non-dimensionalised values here

    cell_pressures = mechanics.get_cell_pressures(areas_nd)
    cell_tensions = mechanics.get_cell_tensions(Gamma, perimeters_nd, L_0)
    P_eff=mechanics.get_P_eff(areas_nd, Gamma, perimeters_nd, L_0)
    cell_shear_stress=((perimeters_nd*cell_tensions)/areas_nd)*zeta
    stress_angle=mechanics.get_stress_angle(P_eff, long_axis_angle)
    edge_tensions=np.array([np.sum(cell_tensions[np.where(B[:, x]!=0)]) for x in range(len(A))])
    #########################
    # Data output
    #########################
    fileio.write_parameters(data_dir,edges_name,stretch_type,t,pixel_size, micron_size,Gamma, Lambda, A_0dim, area_scaling)

    ##Cell Data

    #Only geometric data (no Gamma or Lambda dependence)
    geom_data_names=['cell_id', 'cell_area_microns', 'cell_perimeter_microns', 'cell_centre_x', 'cell_centre_y','shape_parameter', 'circularity', 'cell_edge_count', \
    'long_axis_angle', 'cell_zeta']
    cell_data_geom=np.transpose(np.vstack((cell_id, cell_areas, cell_perimeters,cell_centres[:,0], cell_centres[:,1],\
    shape_parameter, circularity, cell_edge_count, long_axis_angle, zeta)))

    geom_df=pd.DataFrame(cell_data_geom, columns=geom_data_names)

    
    geom_df['time']=t_min
    geom_df['experiment']=exp_id
    geom_df['stretch_type']=stretch_type

    #write geometry data frames and summary stats
    geom_df.to_csv(data_dir + '/'+exp_id+  "_fr%03d"%frame +'_cell_data_geometry.csv', index=False)
    geom_df.iloc[:,1:].describe().to_csv(data_dir + '/'+exp_id+'_geometry_summary_stats.csv')


    all_df=geom_df
    all_df['cell_area_nd']=areas_nd
    all_df['cell_perimeter_nd']=perimeters_nd
    all_df['cell_pressure']=cell_pressures
    all_df['cell_tension']=cell_tensions
    all_df['P_eff']=P_eff
    all_df['shear_stress']=cell_shear_stress
    all_df['stress_angle']=stress_angle


    #write cell data frames and summary stats
    all_df.to_csv(data_dir + '/'+exp_id+  "_fr%03d"%frame +'_cell_data.csv', index=False)
    all_df.iloc[:,1:].describe().to_csv(data_dir + '/'+exp_id+'_summary_stats.csv')

    ##Edge Data

    edge_data_names=['edge_length_microns', 'edge_length_nd', 'edge_tension', 'edge_angle']
    edge_data=np.transpose(np.vstack((edge_lengths, edge_lengths_nd, edge_tensions, edge_angles)))

    edge_df=pd.DataFrame(edge_data, columns=edge_data_names)
    edge_df.to_csv(data_dir + '/'+exp_id+  "_fr%03d"%frame +'_edge_data.csv', index=False)


    #########################
    # Plots
    #########################
    #Comment out or add plots as required.

    """
    Options:

    continuous or discrete plot:

    data: dataframe, all_df or geom_df
    plot_variable: string, name of column in dataframe to plot e.g. 'circularity', 'cell_area_microns'
    plot_label: string, label to appear on the plot
    cmap: string or cm object, the colormap to use in plot
    cc_flag: boolean, 1 to plot cell centres, 0 otherwise
    cell_id_flag: boolean, 1 to plot cell id, 0 otherwise
    shape_angle_flag: boolean, 1 to plot shape axis, 0 otherwise
    stress_angle_flag: boolean, 1 to plot stress axis, 0 otherwise
    
    binary plot:

    data: dataframe, all_df or geom_df
    plot_variable: string, name of column in dataframe to plot e.g. 'circularity', 'cell_area_microns'
    plot_label: string, label to appear on the plot
    threshold: float, number where colours switch e.g. 0 for P_eff
    low_colour: string, colour of cells below threshold
    high_colour: string, colour of cells above threshold
    cc_flag: boolean, 1 to plot cell centres, 0 otherwise
    cell_id_flag: boolean, 1 to plot cell id, 0 otherwise
    shape_angle_flag: boolean, 1 to plot shape axis, 0 otherwise
    stress_angle_flag: boolean, 1 to plot stress axis, 0 otherwise

    """
    visualise.cell_plot_continuous(all_df,'cell_tension','Tension', 'Reds', 0,0,0,0, C, R, cell_centres, edges_name, plot_dir)
    visualise.cell_plot_continuous(all_df,'cell_pressure','Pressure', 'Blues_r', 0,0,0,0, C, R, cell_centres, edges_name, plot_dir)
    visualise.cell_plot_continuous(all_df,'cell_area_microns',r'Area $(\mu \mathrm{m})$', 'Greens', 0,0,0,0, C, R, cell_centres, edges_name, plot_dir)
    visualise.cell_plot_continuous(all_df,'shear_stress','Shear Stress Magnitude', 'plasma', 0,0,0,0, C, R, cell_centres, edges_name, plot_dir)
    visualise.cell_plot_continuous(all_df,'circularity','Circularity', 'viridis', 0,0,1,0, C, R, cell_centres, edges_name, plot_dir)


    visualise.cell_plot_discrete(all_df,'cell_edge_count', 'Number of sides', cm.jet, 0,0,0,0, C, R, cell_centres, edges_name, plot_dir)
    
    visualise.cell_plot_binary(all_df,'P_eff', 'Effective Pressure',0, 'blue', 'red', 0,0,0,1, C, R, cell_centres, edges_name, plot_dir)
    
    visualise.cell_plot_ids(t,C,R,cell_centres, edges_name, plot_dir)

    visualise.edge_plot_continuous(all_df,edge_tensions,'Edge Tension', 'rainbow', 0,0,0,0, A, R, cell_centres, edges_name, plot_dir)

    visualise.angle_hist(edge_angles, 'Edge angles', plot_dir, 12, 180, edges_name)
    visualise.angle_hist(all_df['long_axis_angle'], 'Long axis angles', plot_dir, 12, 180, edges_name)
    visualise.angle_hist(all_df['stress_angle'], 'Major stress axis angles', plot_dir, 12, 180, edges_name)

    visualise.plot_cell_sides(all_df, 'Number of Sides', plot_dir, edges_name)

    visualise.plot_summary_hist(all_df, 'summary_histograms', plot_dir, edges_name)
