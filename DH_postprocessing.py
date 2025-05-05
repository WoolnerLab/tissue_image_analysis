""" 
Dionn Hargreaves

script to read in data and create graphs

e.g. using nearest neighbour shells

"""

import os
import numpy as np
import pandas as pd
import seaborn as sb
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sb

from src import fileio
from src import visualise




config_files = sorted(glob('Input/'+'*_conf.csv'))[:]

all_angles_from_centre = []
all_areas_from_centre = [] 

all_shells = []

all_divs_per_shell = []
all_divs_per_shellarea = []


for config_file in config_files:
    f = open(config_file,'r')
    lines = f.read().splitlines()[1:] ### I'm assuming these are experiment names at different times
    f.close()


    edges_name,t_min, pixel_size, micron_size = fileio.read_conf_line(lines[0])
    exp_id=edges_name.split('_')[0]+'_'+edges_name.split('_')[1]+'_'+edges_name.split('_')[2]
    frame=int(edges_name.split('_')[7][2:])
    initial_path="Output/"+edges_name.split('_trace')[0]
    experiment = exp_id+"_fr%03d"%frame

    experiment_path = sorted(glob(initial_path+"/2025-*"))[-1]

    print(experiment_path)
    shells = np.loadtxt(experiment_path+"/Matrices/"+experiment+"nn_shells.txt")
    A = np.loadtxt(experiment_path+"/Matrices/"+experiment+"_Matrix_A.txt") # edge -> vertex
    B = np.loadtxt(experiment_path+"/Matrices/"+experiment+"_Matrix_B.txt") # cell -> edge
    C = np.loadtxt(experiment_path+"/Matrices/"+experiment+"_Matrix_C.txt") # 
    R = np.loadtxt(experiment_path+"/Matrices/"+experiment+"_Matrix_R.txt") # coordinates of vertices

    scale = pixel_size/micron_size
    geom_df = pd.read_csv(experiment_path+"/Data/"+experiment+"_cell_data_geometry.csv")
    all_df = pd.read_csv(experiment_path+"/Data/"+experiment+"_cell_data.csv")

    cell_centres = np.zeros((len(geom_df.cell_centre_x_microns),2))
    cell_centres[:,0] = scale*geom_df.cell_centre_x_microns
    cell_centres[:,1] = scale*geom_df.cell_centre_y_microns

    approx_centre_of_cap = [np.mean(cell_centres[:,0]), np.mean(cell_centres[:,1])]
    all_cell_distance_from_approx_centre = np.sqrt(((cell_centres - approx_centre_of_cap)*(cell_centres - approx_centre_of_cap)).sum(axis=1))
    find_central_cell = all_cell_distance_from_approx_centre.argsort()[0]

    Use_binary = np.zeros(len(cell_centres[:,0]))
    Use_binary[find_central_cell] = 1
    geom_df['Use_In_Binary'] = Use_binary
    visualise.cell_plot_binary(geom_df,'Use_In_Binary', 'Centre cell',0.5, 'magenta', 'green', 1,0,0,0, C, R, cell_centres, experiment, experiment_path)




    ### we have officially identified the centre cell. Now we can look at this cell's neighbours: 
    geom_df['Neighbours'] = np.where(shells[find_central_cell,:]==1, shells[find_central_cell,:], 0) # immediate neighbours
    geom_df['Shells'] = shells[find_central_cell,:]

    geom_df['Shells']  = [int(x) for x in geom_df['Shells']]
    ## check centre cell immediate neighbours, and all shells as examples
    visualise.cell_plot_binary(geom_df,'Neighbours', 'Centre cell shell',0.5, 'magenta', 'green', 1,0,0,0, C, R, cell_centres, experiment, experiment_path)
    visualise.cell_plot_continuous(geom_df,'Shells','Shells', 'Set3', 0,0,0,0, C, R, cell_centres, experiment, experiment_path)
    visualise.cell_plot_continuous(all_df,'P_eff','P_eff', 'bwr', 0,0,0,0, C, R, cell_centres, experiment, experiment_path)

    #

    visualise.angle_hist(geom_df.long_axis_angle*180/np.pi, 'Long axis angle', experiment_path, 9, 180, experiment)


    ### add pi/2 to cell centres arctan and mod pi

    ## reframe all cells to central cell
    reframed_cells = cell_centres-cell_centres[find_central_cell,:]
    cell_angles_from_centre = (np.arctan2(reframed_cells[:,1], reframed_cells[:,0]) ) % np.pi


    average_cell_size = np.mean(geom_df.cell_area_microns)
    nondim_cell_size = geom_df.cell_area_microns/average_cell_size
    ### cell elongation angle relative to cap central cell
    
    angle_differences = ((geom_df.long_axis_angle)-cell_angles_from_centre) #% np.pi/2
    geom_df['Alignment with radius'] = np.cos(angle_differences)**2

    visualise.cell_plot_continuous(geom_df,'Alignment with radius','Alignment with radius', 'bwr', 0,0,1,0, C, R, cell_centres, experiment, experiment_path)


    shell_population_angles = []
    division_cell_count = []
    no_in_shell = []
    avg_alignment_to_centre = []
    avg_alignment_to_centre_SD = []
    avg_alignment_to_centre_SEM = []
    avg_cell_area = []
    avg_cell_area_SEM = []
    bulk_pressures = []
    cell_areas = []
    avg_bulk_pressure = []
    avg_cell_circ = []
    avg_cell_circ_SEM = []

    for shell_no in range(int(max(geom_df.Shells))+1):
        current_cell_areas = geom_df.cell_area_microns[np.where(shells[find_central_cell,:]==shell_no)[0]]
        avg_cell_circ.append([np.mean(geom_df.circularity[np.where(shells[find_central_cell,:]==shell_no)[0]])])
        avg_cell_circ_SEM.append([np.std(geom_df.circularity[np.where(shells[find_central_cell,:]==shell_no)[0]])/np.sqrt(len(current_cell_areas))])
        #current_P_eff = all_df.P_eff[np.where(shells[find_central_cell,:]==shell_no)[0]]
        avg_cell_area.extend([np.mean(current_cell_areas)])
        avg_cell_area_SEM.extend([np.std(current_cell_areas)/np.sqrt(len(current_cell_areas))])
        current_cell_angles = angle_differences[np.where(shells[find_central_cell,:]==shell_no)[0]]
        mean_current_angles = np.mean(np.cos(current_cell_angles)**2)
        avg_alignment_to_centre.extend([mean_current_angles])
        avg_alignment_to_centre_SD.extend([np.std(np.cos(current_cell_angles)**2)])
        avg_alignment_to_centre_SEM.extend([np.std(np.cos(current_cell_angles)**2)/np.sqrt(len(current_cell_angles))])
        shell_population_angles.append(current_cell_angles)
        dividing_count = sum(geom_df.division_flag[np.where(shells[find_central_cell,:]==shell_no)[0]])
        no_in_shell.extend([len(np.where(shells[find_central_cell,:]==shell_no)[0])])
        division_cell_count.extend([dividing_count])
        #cell_areas.extend(current_cell_areas)
        #bulk_pressures.extend(current_P_eff*current_cell_areas)
        avg_bulk_pressure.append(sum(all_df.cell_area_nd[np.where(shells[find_central_cell,:]<=shell_no)[0]]*all_df.P_eff[np.where(shells[find_central_cell,:]<=shell_no)[0]])/sum(all_df.cell_area_nd[np.where(shells[find_central_cell,:]<=shell_no)[0]]))

    np.savetxt(experiment_path+'/bulk_pressure.txt', avg_bulk_pressure)
    division_cell_per_shell = np.asarray(division_cell_count)/np.asarray(no_in_shell)
    division_cell_per_shellarea = np.cumsum(division_cell_count)/np.cumsum(no_in_shell)

    geom_df['Non-dim cell area'] = nondim_cell_size


    sb.pointplot(data=geom_df, x='Shells', y='Alignment with radius', errorbar = "se", linestyle = "none", capsize = .2, color='black', markersize = 3, err_kws={'linewidth': 1}) ## currently between 0 and 180, make sure it's between 0 and 90 forward
    plt.savefig(experiment_path+"/Radial alignment.png")
    plt.close()
    sb.pointplot(data=geom_df, x='Shells', y='Non-dim cell area', errorbar = "se", linestyle = "none", capsize = .2, color='black', markersize = 3, err_kws={'linewidth': 1})  
    plt.savefig(experiment_path+"/cell area.png")     
    plt.close()  

    plt.plot(range(int(max(geom_df.Shells))+1),division_cell_per_shellarea, color='black') 
    plt.savefig(experiment_path+"/divisions per area.png")
    plt.close()

    all_shells.extend(geom_df.Shells)
    all_angles_from_centre.extend(angle_differences)
    all_areas_from_centre.extend(nondim_cell_size)
    np.savetxt(experiment_path+'/Counts_shells.txt', division_cell_per_shell)
    np.savetxt(experiment_path+'/Counts_shellsarea.txt', division_cell_per_shellarea)

    np.savetxt(experiment_path+'/Avg_alignment_per_shell.txt', avg_alignment_to_centre)
    np.savetxt(experiment_path+'/Avg_alignment_per_shell_SD.txt', avg_alignment_to_centre_SD)
    np.savetxt(experiment_path+'/Avg_alignment_per_shell_SEM.txt', avg_alignment_to_centre_SEM)
    np.savetxt(experiment_path+'/Avg_cell_area.txt', avg_cell_area)
    np.savetxt(experiment_path+'/Avg_cell_area_SEM.txt', avg_cell_area_SEM)
    np.savetxt(experiment_path+'/Avg_cell_circ.txt', avg_cell_circ)
    np.savetxt(experiment_path+'/Avg_cell_circ_SEM.txt', avg_cell_circ_SEM)



    ##### we want to take a subset of the 'shells' matrix; i.e. only take the rows and columns of dividing cells
    shells_dividing_subset = np.zeros(int((len(np.where(geom_df.division_flag == 1)[0])**2-len(np.where(geom_df.division_flag == 1)[0]))/2))

    count = 0
    for i in range(len(np.where(geom_df.division_flag == 1)[0])):
        for j in range(i+1,len(np.where(geom_df.division_flag == 1)[0])):
            shells_dividing_subset[count] = shells[np.where(geom_df.division_flag == 1)[0][i], np.where(geom_df.division_flag == 1)[0][j]]
            count+=1
    

    expectation_coefficient = (len(np.where(geom_df.division_flag==1)[0])/len(geom_df.division_flag))*(len(np.where(geom_df.division_flag==1)[0]-1)/(len(geom_df.division_flag)-1))

    pcf = np.zeros(int(max(np.ravel(shells))))
    for m in range(int(max(np.ravel(shells)))):
        pcf[m] = len(np.where(shells_dividing_subset == m)[0])/(expectation_coefficient*0.5*len(np.where(np.ravel(shells)==m)[0]))
    np.savetxt(experiment_path+'/pcf_2D.txt', pcf)


    plt.plot(pcf)
    plt.savefig(experiment_path+'/2D_PCF.png', dpi=500)
    
    cap_density = len(geom_df.cell_area_microns)/np.sum(geom_df.cell_area_microns)
    np.savetxt(experiment_path+'/cap_density.txt', [cap_density])


net_df = pd.DataFrame({'Shells': all_shells,
     'Angles': np.cos(all_angles_from_centre)**2,
     'Areas': all_areas_from_centre
    })



sb.pointplot(data=net_df, x='Shells', y='Angles', linestyle = "none", errorbar = "se", capsize = .2, color='black', markersize = 3, err_kws={'linewidth': 1})   ## currently between 0 and 180, make sure it's between 0 and 90 forward
plt.savefig("Output/Radial alignment.png", dpi = 500)
plt.close()
sb.pointplot(data=net_df, x='Shells', y='Areas', linestyle = "none", errorbar = "se", capsize = .2, color='black', markersize = 3, err_kws={'linewidth': 1}) 
plt.savefig("Output/cell area.png", dpi = 500)     
plt.close()  

sb.histplot(data=net_df, x='Angles')

#plt.savefig("Output/divisions per area.png")



#### Need to save the above globally for whole stats analysis. 
###