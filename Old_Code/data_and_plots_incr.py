"""
Code to generate matrices from just Matrices

"""

import os
import numpy as np
import pandas as pd

from datetime import datetime
from glob import glob

from utils import fileio
from utils import matrices
from utils import geometry
from utils import mechanics
from . import visualise


#establish directory structure
CURRENT_DIR = os.getcwd()
input_dir=CURRENT_DIR+'/Input/'
output_dir='C:\\Users\\v35431nc\\Documents\\Lab_Stuff\\Relaxation\\20230203_incremental/'

#########################
#User Input
#########################


#Path to directory where Matrices are stored
indata_dir="C:\\Users\\v35431nc\\Documents\\Lab_Stuff\\Relaxation\\20230203_incremental\\Matrices"

#########################
#Constant variables
#########################

Lambda = -0.259 # Line tension (tunes P_0) (non-dimensional)
Gamma = 0.172  # Contractility (non-dimensional)
pref_perimeter  = -Lambda/(2*Gamma)  # Cell preferred perimeter (non-dimensional)
area_scaling = 0.00001785     # Gradient of the change in prefArea, used for time scaling. NB: if running code for knock downs you may need to recalculate this number.
t_relax = 20.0 #computational parameter not important for image analysis                                  # not needed, take out if want to
ExperimentFlag = 1   

#########################
#Setup directories and load Matrices
#########################
c_files=sorted(glob(input_dir+'/20230203_1_IN_BFPCAAX-CheHis_us_*_SP_fr*_trace_conf.csv'))
for f in c_files:
    #read in conf file
    edges_name,t_min, pixel_size, micron_size = fileio.read_conf(f)
    exp_id=edges_name.split('_')[0]+'_'+edges_name.split('_')[1]+'_'+edges_name.split('_')[4]+'_'+edges_name.split('_')[5]+'_'+edges_name.split('_')[7]
    print(exp_id)
    t=t_min*60.0
    stretch_type=edges_name.split('_')[4][-1]

    #make directories to output to
    if os.path.exists(output_dir+edges_name.split('_trace')[0])==False: os.mkdir(output_dir+edges_name.split('_trace')[0])

    mydir = os.path.join(output_dir+edges_name.split('_trace')[0], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(mydir)
    data_dir=mydir+"/Data"
    plot_dir=mydir+"/Plots"
    if os.path.exists(data_dir)==False: os.mkdir(data_dir)
    if os.path.exists(plot_dir)==False: os.mkdir(plot_dir)



    A  = np.loadtxt(glob(indata_dir+'/'+exp_id+'*Matrix_A.txt')[0]) # Incidence matrix. Rows => edges; columns => vertices.
    B  = np.loadtxt(glob(indata_dir+'/'+exp_id+'*Matrix_B.txt')[0]) # Incidence matrix. Rows => cells; columns => edges. Values +/-1 for orientation
    C  = np.loadtxt(glob(indata_dir+'/'+exp_id+'*Matrix_C.txt')[0]) # Incidence matrix. Rows => cells; columns => vertices. 
    R  = np.loadtxt(glob(indata_dir+'/'+exp_id+'*Matrix_R.txt')[0]) # Coordinates of vertices

    ##############################
    #Get cell geometry
    ##############################
    R=R*(micron_size/pixel_size)

    cell_areas=geometry.get_areas(A,B, R)
    #print(cell_areas)
    cell_perimeters=geometry.get_perimeters(A,B,R)
    cell_edge_count=geometry.get_edge_count(B)
    cell_centres=geometry.get_cell_centres(C,R,cell_edge_count)
    mean_cell_area=geometry.get_mean_area(cell_areas)      
    print("Mean area = ", mean_cell_area)

    ##############################
    #Find prefered area
    ##############################

    #if experiment is unstretched or we are sat time=0, then we calculate the preffered area, otherwise read from file.
    if stretch_type=='u' or t==0.0:

        #Root finding to get prefered area where global stress=0
        pref_area=geometry.get_pref_area(cell_areas, Gamma, cell_perimeters, pref_perimeter, mean_cell_area)
        print("Prefered area = ", pref_area)
        fileio.write_pref_area(data_dir,input_dir, edges_name, pref_area)
    else:
        #if not t=0 or unstretched then read in the pref_area (needs to be copied to input with traces)
        pref_area=fileio.read_pref_area(input_dir, edges_name)


    fileio.write_parameters(mydir,edges_name,stretch_type,t,pixel_size, micron_size,Gamma, Lambda, pref_area, area_scaling)

    ##############################
    #Non-dimensionalisation
    ##############################

    t_nd=t/t_relax ##possibly superflus here (only needed if analysing simulation data).

    #if cells are stretched then we need to use a scaling factor on the prefered area. 
    if stretch_type != 0:
        pref_area *=(1-area_scaling*t)

    #non-dimensionalise spatial data
    R_nd=R/((pref_area)**0.5)
    areas_nd=cell_areas/pref_area
    perimeters_nd=cell_perimeters/((pref_area)**0.5)
    tangents_nd=geometry.get_tangents(A,R_nd)
    edge_lengths_nd=geometry.get_edge_lengths(tangents_nd)
    cell_centres_nd=geometry.get_cell_centres(C,R_nd,cell_edge_count)
    cell_centres_nd=geometry.get_cell_centres(C,R_nd,cell_edge_count)

    ##############################
    #Generate data
    ##############################
    print("Generating Data")

    N_c=np.shape(C)[0]
    cell_id=np.linspace(0, N_c-1,N_c)
    cell_P_eff = mechanics.get_P_eff(areas_nd, Gamma, perimeters_nd, pref_perimeter) 
    cell_pressures = mechanics.get_cell_pressures(areas_nd)
    cell_tensions = mechanics.get_cell_tensions(Gamma, perimeters_nd, pref_perimeter)
    cell_shears,cell_zetas = mechanics.calc_shear(tangents_nd,edge_lengths_nd,B,perimeters_nd,cell_tensions,areas_nd)
    cell_circularity, major_shape_axis, major_shape_axis_alignment,major_stress_axis, major_stress_axis_alignment=geometry.get_shape_tensor(R_nd,C,cell_edge_count,cell_centres_nd,cell_P_eff)
    shape_parameter = perimeters_nd/(np.sqrt(areas_nd))

    global_stress= mechanics.get_global_stress(cell_P_eff, areas_nd)
    monolayer_energy=mechanics.get_monolayer_energy(areas_nd, perimeters_nd, pref_perimeter)



    #combine data into a dataframe object
    data_names=['cell_id', 'cell_perimeter_microns', 'cell_area_microns', 'cell_perimeter_nd',\
    'cell_area_nd', 'cell_P_eff_nd', 'shape_parameter', 'circularity', 'cell_edge_count', \
    'cell_shear', 'cell_zeta', 'major_shape_axis_alignment_rads', 'major_stress_axis_alignment_rads']
    cell_data_all=np.transpose(np.vstack((cell_id, cell_perimeters, cell_areas, perimeters_nd, areas_nd,\
    cell_P_eff, shape_parameter, cell_circularity, cell_edge_count, cell_shears, cell_zetas, major_shape_axis_alignment, major_stress_axis_alignment)))

    cell_df=pd.DataFrame(cell_data_all, columns=data_names) 

    #Only geometric data (no Gamma or Lambda dependence)
    geom_data_names=['cell_id', 'cell_perimeter_microns', 'cell_area_microns', 'shape_parameter', 'circularity', 'cell_edge_count', \
    'major_shape_axis_alignment_rads']
    cell_data_geom=np.transpose(np.vstack((cell_id, cell_perimeters, cell_areas,\
    shape_parameter, cell_circularity, cell_edge_count, major_shape_axis_alignment)))

    geom_df=pd.DataFrame(cell_data_geom, columns=geom_data_names)

    #write data frames and summary stats
    #cell_df.to_csv(data_dir + '/'+exp_id+'_cell_data_all_Gamma_'+str(Gamma)+'_Lambda_'+str(Lambda)+'.csv', index=False)
    #cell_df.iloc[:,1:].describe().to_csv(data_dir + '/'+exp_id+'_summary_stats_Gamma_'+str(Gamma)+'_Lambda_'+str(Lambda)+'.csv')

    geom_df.to_csv(data_dir + '/'+exp_id+'_cell_data_geometry.csv', index=False)
    geom_df.iloc[:,1:].describe().to_csv(data_dir + '/'+exp_id+'_geometry_summary_stats.csv')

    #fileio.write_global_data(global_stress, monolayer_energy, data_dir,edges_name) #write global data

    axisLength = micron_size/(pref_area**(1/2.)) + 0.5
    visualise.plot_summary_hist(geom_df,'geom_data', plot_dir, edges_name)
    #discrete data
    visualise.plot_cell_sides(geom_df, "Number_of_Sides", plot_dir, edges_name)

    #angle histogram

    visualise.angle_hist(geom_df['major_shape_axis_alignment_rads'], "Major Shape Axis Alignment", plot_dir, 12, 180 , edges_name)
    
    visualise.graphNetworkColourBinary('Effective Pressure Binary',cell_P_eff,'red','blue',0,1,0,\
    t,A,C,R,cell_centres,cell_P_eff,major_stress_axis,axisLength,plot_dir,edges_name,ExperimentFlag,0, 'black', 'png')

    
    visualise.graphNetworkColourBinary('Cell_id',cell_df['cell_id'],'black','black',0.0,0,0,t,A,C,R,cell_centres,cell_P_eff,major_stress_axis,axisLength,plot_dir,edges_name,ExperimentFlag, 1, 'blue','png')

    visualise.graphNetworkColorBar('Circularity',cell_circularity,'viridis',1,0.0,1.0, t,A,C,R,cell_centres,axisLength,plot_dir,edges_name,ExperimentFlag,0,'png')