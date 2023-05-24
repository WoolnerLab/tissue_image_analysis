"""
Created on 20/01/2023
Trace analysis



@author: Natasha Cowley
Contributors:
Emma Johns
Alex Nestor-Bergmann - image detection code
"""
import glob
import os

import numpy as np
import pandas as pd

from datetime import datetime


from source import tissue
from source import cell
from source import edge_detection
from source import setup_points
from source import graham_scan

from utils import fileio
from utils import handtrace
from utils import matrices
from utils import geometry
from utils import mechanics
from utils import visualise

#establish directory structure
CURRENT_DIR = os.getcwd()
input_dir=CURRENT_DIR+'/Input/'
output_dir=CURRENT_DIR+'/Output/'

#########################
#User Input
#########################

conf_file=input_dir+'20230426_2_NT_GFPTub-ControlMO_bf_0p5_MP_fr1_trace_junctions.csv' #name of config file

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
#Setup directories
#########################

#read in conf file
edges_name,t_min, pixel_size, micron_size = fileio.read_conf(conf_file)
exp_id=edges_name.split('_')[0]+'_'+edges_name.split('_')[1]+'_'+edges_name.split('_')[7]
t=t_min*60.0
stretch_type=edges_name.split('_')[3][-1]

#make directories to output to

save_dir, trace_dir, matrix_dir, data_dir, plot_dir = fileio.setup_directories(output_dir, edges_name)

##############################
#Run traces and construct matrices
##############################
print("Extracing trace data")
edge_verts,cells, cell_edges, A, B, C, R, image0 = handtrace.run_trace(edges_name, input_dir)
fileio.write_cell_data(trace_dir,edge_verts,cells, cell_edges, exp_id)
handtrace.check_trace_plot(save_dir,image0, cells, R, edges_name)
print("Writing Matrices")
fileio.write_matrices(matrix_dir, A, B, C, R, exp_id)

##############################
#Get cell geometry
##############################

#make sure pixel size is size of tif not of trace as some have borders
R=R*(micron_size/pixel_size)

cell_areas=geometry.get_areas(A,B, R)
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


fileio.write_parameters(save_dir,edges_name,stretch_type,t,pixel_size, micron_size,Gamma, Lambda, pref_area, area_scaling)
##############################
#Non-dimensionalisation
##############################

t_nd=t/t_relax ##possibly superflus here.

#if cells are stretched then we need to use a scaling factor on the prefered area. 
if stretch_type != 'u':
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
cell_df.to_csv(data_dir + '/'+exp_id+'_cell_data_all_Gamma_'+str(Gamma)+'_Lambda_'+str(Lambda)+'.csv', index=False)
cell_df.iloc[:,1:].describe().to_csv(data_dir + '/'+exp_id+'_summary_stats_Gamma_'+str(Gamma)+'_Lambda_'+str(Lambda)+'.csv')

geom_df.to_csv(data_dir + '/'+exp_id+'_cell_data_geometry.csv', index=False)
geom_df.iloc[:,1:].describe().to_csv(data_dir + '/'+exp_id+'_geometry_summary_stats.csv')

fileio.write_global_data(global_stress, monolayer_energy, data_dir,edges_name) #write global data



##############################
#Plot
##############################
print("Generating Plots")
### Distributions ###

#Summary Histograms for continuous data 
visualise.plot_summary_hist(cell_df,'all_data', plot_dir, edges_name)
visualise.plot_summary_hist(geom_df,'geom_data', plot_dir, edges_name)

#discrete data
visualise.plot_cell_sides(cell_df, "Number_of_Sides", plot_dir, edges_name)

#angle histogram
visualise.angle_hist(cell_df['major_shape_axis_alignment_rads'], "Major Shape Axis Alignment", plot_dir, 6, 90 , edges_name)
visualise.angle_hist(cell_df['major_stress_axis_alignment_rads'], "Major Stress Axis Alignment", plot_dir, 6, 90 , edges_name)
##### Network Plots ####
axisLength = micron_size/(pref_area**(1/2.)) + 0.5

### Gradient plots ###
#numbered plot for reference

visualise.graphNetworkColorBar('Cell Areas Numbered for Reference',cell_areas,'Greens',0,0,1400,\
t,A,C,R,cell_centres,axisLength,plot_dir,edges_name,ExperimentFlag,1, 'pdf')
# Area #
visualise.graphNetworkColorBar('Cell Areas',cell_areas,'Greens',0,0,1400,\
t,A,C,R,cell_centres,axisLength,plot_dir,edges_name,ExperimentFlag,0, 'png')

# # Number of sides #  #NB colorbar will need adjusting for any oficial figures
# visualise.graphNetworkColorBar('Number of Sides',cell_edge_count,'Paired',1,4.0,11.0,\
# t,A,C,R,cell_centres,axisLength,plot_dir,edges_name,ExperimentFlag,0, 'png')

# # Circularity #
# visualise.graphNetworkColorBar('Circularity',cell_circularity,'viridis',1,0.0,1.0,\
# t,A,C,R,cell_centres,axisLength,plot_dir,edges_name,ExperimentFlag,0, 'png')

# visualise.graphNetworkColorBar('Elongation',1.0-cell_circularity,'viridis',1,0.0,1.0,\
# t,A,C,R,cell_centres,axisLength,plot_dir,edges_name,ExperimentFlag,0, 'png')

# # Shear Strain #
# visualise.graphNetworkColorBar('Shear Strain',cell_zetas,'viridis',-1,0.0,0.35,\
# t,A,C,R,cell_centres,axisLength,plot_dir,edges_name,ExperimentFlag,0, 'png')

# # Cell effective pressures # (bare in mind forcing limits maybe necessary for nice symmetry round 0)
# visualise.graphNetworkColorBar('Absolute Effective Pressure',abs(cell_P_eff),'plasma',1,0.0,0.5,\
# t,A,C,R,cell_centres,axisLength,plot_dir,edges_name,ExperimentFlag,0, 'png')

# visualise.graphNetworkColorBar('Effective Pressure',cell_P_eff,'bwr',1,-1.0,1.0,\
# t,A,C,R,cell_centres,axisLength,plot_dir,edges_name,ExperimentFlag,0, 'png')

# # Shear Stress
# visualise.graphNetworkColorBar('Shear Stress',cell_shears,'plasma',-1,0.0,0.7,\
# t,A,C,R,cell_centres,axisLength,plot_dir,edges_name,ExperimentFlag,0, 'png')

# # Cell Tension and Pressure
# visualise.graphNetworkColorBar('Cell Tension',cell_tensions,'inferno',-1,0.0,0.5,\
# t,A,C,R,cell_centres,axisLength,plot_dir,edges_name,ExperimentFlag,0, 'png')

# visualise.graphNetworkColorBar('Cell Pressure',cell_pressures,'cividis',-1,0.0,0.5,\
# t,A,C,R,cell_centres,axisLength,plot_dir,edges_name,ExperimentFlag,0, 'png')

# #### Binary Plots ###

# # Areas #
# visualise.graphNetworkColourBinary('Cell Areas Binary',cell_areas,'darkgreen','palegreen',np.median(cell_areas),0,0,\
# t,A,C,R,cell_centres,cell_P_eff,major_stress_axis,axisLength,plot_dir,edges_name,ExperimentFlag, 'png')

# #Number of sides
# visualise.graphNetworkColourBinary('Number Of Sides Binary',cell_edge_count,'darkorchid','thistle',np.median(cell_edge_count),0,1,\
# t,A,C,R,cell_centres,cell_P_eff,major_stress_axis,axisLength,plot_dir,edges_name,ExperimentFlag, 'png')

# #Circularity#
# visualise.graphNetworkColourBinary('Cell Circularity Binary',cell_circularity,'darkorange','bisque',2.0/3.0,0,0,\
# t,A,C,R,cell_centres,cell_P_eff,major_stress_axis,axisLength,plot_dir,edges_name,ExperimentFlag, 'png')

# visualise.graphNetworkColourBinary('Cell Elongation Binary',1.0-cell_circularity,'crimson','pink',2.0/3.0,0,0,\
# t,A,C,R,cell_centres,cell_P_eff,major_stress_axis,axisLength,plot_dir,edges_name,ExperimentFlag, 'png')

# # Shear Strain
# visualise.graphNetworkColourBinary('Shear strain Binary',cell_zetas,'hotpink','gold',np.median(cell_zetas),1,0,\
# t,A,C,R,cell_centres,cell_P_eff,major_stress_axis,axisLength,plot_dir,edges_name,ExperimentFlag, 'png')

# # Effective pressure
# visualise.graphNetworkColourBinary('Magnitude of effective pressure Binary',abs(cell_P_eff),'yellow','lemonchiffon',np.median(abs(cell_P_eff)),1,0,\
# t,A,C,R,cell_centres,cell_P_eff,major_stress_axis,axisLength,plot_dir,edges_name,ExperimentFlag, 'png')

# visualise.graphNetworkColourBinary('Effective Pressure Binary',cell_P_eff,'red','blue',0,1,0,\
# t,A,C,R,cell_centres,cell_P_eff,major_stress_axis,axisLength,plot_dir,edges_name,ExperimentFlag, 'png')

# #Shear Stress
# visualise.graphNetworkColourBinary('Cell Shears Binary',cell_shears,'darkseagreen','honeydew',np.mean(cell_shears),0,0,\
# t,A,C,R,cell_centres,cell_P_eff,major_stress_axis,axisLength,plot_dir,edges_name,ExperimentFlag, 'png')