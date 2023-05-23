"""
visualise.py

Functions to plot data

Graph network plots are adapted from code written by Emma Johns

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection


def make_polygon(i, C, R, cell_centres):
    """
    Generate polygon

    Parameters:
    i (int): cell id
    C (numpy array): Nc x Nv order array relating cells to vertices
    R (numpy array): vertex coordinates
    cell_centres (numpy array): cell centre coordinates
    """

    Ralpha=R[np.where(C[i,:]==1)[0]]-cell_centres[i] #ref frame of cell
    ang=np.arctan2(Ralpha[:,1], Ralpha[:,0])%(2*np.pi) #find angle with x axis
    R_ang=np.transpose(np.vstack((np.where(C[i,:]==1)[0], ang))) #stack index of vertices with angle
    ordered_vertices=R_ang[np.argsort(R_ang[:,-1], axis=0)] #sort by anticlockwise angle
    polygon = Polygon(R[ordered_vertices[:,0].astype(int)],closed = True)
    return polygon


def graphNetworkColorBar(plot_name,plot_var,map_name,ColourBarLimitFlag,barMin,barMax,\
t,A,C,R,cell_centres,axisLength,savedir,edges_name,ExperimentFlag, NumberFlag, file_type):
    """
    Plots of cell network where cells are coloured according to some quantity

    Parameters:
    plot_name (string): Plot title and also name of figure.
    plot_var (numpy array): Varible used to colour cells
    map_name (string): name of colormap to use
    ColourBarLimitFlag (bool): if set to 1 then supplied limits used for colorbar
    barMin (float): minimum value for colorbar
    barMax (float): maximum value for colorbar
    t (float): time in experiment
    A (numpy array): Ne x Nv order array relating edges to vertices
    C (numpy array): Nc x Nv order array relating cells to vertices
    R (numpy array): vertex coordinates
    cell_centres (numpy array): cell centre coordinates
    axisLength (float): size of plot
    savedir (string): location to save plot
    edges_name (string): filename of trace
    ExperimentFlag (bool): sets axis length based on experiment or sims (set to 1)
    NumberFlag (bool): If 1 then number of cell will be added to each plot.
    file_type (string): file type to save image as

    """

    N_c=np.shape(C)[0]
    N_e=np.shape(A)[0]

    beg_edge = ((abs(A) - A)*0.5)@R
    end_edge = ((abs(A) + A)*0.5)@R

    fig, ax = plt.subplots()
    patches = []

    for i in range(N_c):
        polygon = make_polygon(i, C, R, cell_centres)
        patches.append(polygon)

    p = PatchCollection(patches,alpha = 1.0)
    p.set_array(plot_var)
    p.set_cmap(map_name) #set polygon colourmap

    if ColourBarLimitFlag ==1:
        p.set_clim(barMin,barMax)

    ax.add_collection(p) #add polygons to plot


    cbar = fig.colorbar(p, ax=ax)
    cbar.ax.set_ylabel(plot_name, rotation=90)


    ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)

   

    #Plot Edges
    for j in range(0,N_e):
        plt.plot([beg_edge[j,0],end_edge[j,0]],[beg_edge[j,1],end_edge[j,1]],'k',alpha=1.0,linestyle ='-')
    if NumberFlag==1:
        for i in range(0,N_c):
            plt.plot(cell_centres[i,0],cell_centres[i,1],'k',marker ='o',markersize=2)
            plt.text(cell_centres[i][0], cell_centres[i][1], str(i),fontsize= 5,color='b')
    #Figure specs
    if ExperimentFlag == 1:
        plt.axes([0,0,axisLength,axisLength])
        plt.xlim(0,axisLength)
        plt.ylim(0,axisLength)
    elif ExperimentFlag == 0:
        plt.xlim(-axisLength,axisLength)
        plt.ylim(-axisLength,axisLength)
    else:
        print('Error in axisLength')
    plt.gca().set_aspect('equal')
    ax.set_title("t = {}s, {}".format(t,plot_name))

    #Save Figure
    plt.savefig(savedir + "/"+edges_name+"_" + plot_name+"."+ file_type)
    plt.close()


def graphNetworkColourBinary(CellPropertyName,CellProperty,ColourHigh,ColourLow,CutPoint,MajorAxisIndicator,CellCentreIndicator,t,A,C,R,cell_centres,cell_P_eff,major_axis,axisLength,savedir,edges_name,ExperimentFlag, file_type):

    """
    Plots of cell network where cells are coloured in a binary fashion
    Optionally major axis and centre dots can be plotted.

    Parameters:
    CellPropertyName (string): Plot title and also name of figure.
    CellProperty (numpy array): Varible used to colour cells
    ColourHigh (string): Colour of high cells (https://matplotlib.org/stable/gallery/color/named_colors.html)
    ColourLow (string): Colour of low cells
    CutPoint (float): Choice of threshold between high and low cells
    MajorAxisIndicator (bool): 1 to plot major axis, 0 for not
    CellCentreIndicator (bool): 1 to plot cell centre, 0 to not 
    t (float): time in experiment
    A (numpy array): Ne x Nv order array relating edges to vertices
    C (numpy array): Nc x Nv order array relating cells to vertices
    R (numpy array): vertex coordinates
    cell_centres (numpy array): cell centre coordinates
    cell_P_eff (numpy array): cell effective pressure
    major_axis (numpy array): cell major axis
    axisLength (float): size of plot
    savedir (string): location to save plot
    edges_name (string): filename of trace
    ExperimentFlag (bool): sets axis length based on experiment or sims (set to 1)
    file_type (string): file type to save image as
    """
    N_c=np.shape(C)[0]
    N_e=np.shape(A)[0]

    beg_edge = ((abs(A) - A)*0.5)@R
    end_edge = ((abs(A) + A)*0.5)@R

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    
    patches = []
    patchesBlue = []
    patchesRed = []

    for i in range(N_c):
        polygon = make_polygon(i, C, R, cell_centres)
        patches.append(polygon)
        
        if CellProperty[i] < CutPoint:
            polygonBlue = polygon
            patchesBlue.append(polygonBlue)
        else:
            polygonRed = polygon
            patchesRed.append(polygonRed)


    ### For binary effective pressure
    p_blue = PatchCollection(patchesBlue,alpha = 0.5)
    p_blue.set_facecolor(ColourLow)
    ax.add_collection(p_blue)
    p_red = PatchCollection(patchesRed,alpha = 0.7)
    p_red.set_facecolor(ColourHigh)
    ax.add_collection(p_red)


    ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)

    #Plot Edges
    for j in range(0,N_e):
        ax.plot([beg_edge[j,0],end_edge[j,0]],[beg_edge[j,1],end_edge[j,1]],'black',alpha=1.0,linestyle ='-',linewidth=1.0)

    #Plot the centre of cell, the EffectivePressure, the cell number
    for i in range(0,N_c):
        if CellCentreIndicator ==1:
            plt.plot(cell_centres[i,0],cell_centres[i,1],'k',marker ='o',markersize=2)
        if MajorAxisIndicator ==1:
            if cell_P_eff[i]<0:
                plt.quiver(cell_centres[i,0],cell_centres[i,1],major_axis[i,0],major_axis[i,1],edgecolor = 'black',scale=110,width=0.002,headwidth=0.0,headlength=0.0,headaxislength=0.0)
                plt.quiver(cell_centres[i,0],cell_centres[i,1],-major_axis[i,0],-major_axis[i,1],edgecolor = 'black',scale=110,width=0.002,headwidth=0.0,headlength=0.0,headaxislength=0.0)
            else:
                plt.quiver(cell_centres[i,0],cell_centres[i,1],major_axis[i,0],major_axis[i,1],facecolor = 'black',scale=110,width=0.002,headwidth=0.0,headlength=0.0,headaxislength=0.0)
                plt.quiver(cell_centres[i,0],cell_centres[i,1],-major_axis[i,0],-major_axis[i,1],facecolor = 'black',scale=110,width=0.002,headwidth=0.0,headlength=0.0,headaxislength=0.0)
 
    
    #Figure specs
    if ExperimentFlag == 1:
        plt.axes([0,0,axisLength,axisLength])
        plt.xlim(0,axisLength)
        plt.ylim(0,axisLength)
    elif ExperimentFlag == 0:
        plt.xlim(-axisLength,axisLength)
        plt.ylim(-axisLength,axisLength)
    else:
        print('Error in axisLength')
    
    ax.set_title("t = {}s, {}".format(t,CellPropertyName))
    #plt.gcf()
    plt.gca().set_aspect('equal')
    #Save Figure
    fig.savefig(savedir + "/"+edges_name+"_" + CellPropertyName+"."+ file_type)

    plt.close()

def angle_hist(plot_val, plot_name, savedir, bins_number, theta_lim, edges_name):
    """
    Function to plot a histogram binned by angle

    Parameters:
    plot_val (numpy array): Varible to be plotted
    plot_name (string): Plot title and also name of figure.
    savedir (string): location to save plot
    bins_number (int): number of bins in histogram
    theta_lim (float): maximum theta for plot, in degrees
    edges_name (string): filename of trace
    """

    # number of equal bins
    bins = np.linspace(0.0, theta_lim/180 *np.pi, bins_number + 1)

    angle=plot_val

    n, _, _ = plt.hist(plot_val, bins)

    plt.clf()
    width = np.pi/2 / bins_number
    ax = plt.subplot(1, 1, 1, projection='polar')
    bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0, align='edge',color='blue', edgecolor='k')
    for bar in bars:
        bar.set_alpha(0.5)
        
    ax.set_xticks(bins)
    ax.set_thetamin(0)
    ax.set_thetamax(theta_lim)
    ax.yaxis.grid(False)
    ax.set_xlabel('Frequency')
    ax.set_title('Major Axis Alignment')
    ax.xaxis.labelpad=20

    plt.savefig(savedir + "/"+edges_name+"_" + plot_name+".png")
    plt.close()

def plot_cell_sides(cell_data, plot_name, savedir, edges_name):
    """
    Function to plot a var chart of the number of sides oer cell

    Parameters:
    cell_data (pandas dataframe): cell based data
    plot_name (string): Plot title and also name of figure.
    savedir (string): location to save plot
    edges_name (string): filename of trace
    """
    countedges=cell_data['cell_edge_count'].value_counts().sort_index()
    cell_sides=np.array(countedges.index).astype(int)
    edge_frequency=np.array(countedges)
    fig, ax = plt.subplots()

    ax.bar(cell_sides, edge_frequency, color=plt.cm.tab20(cell_sides))

    ax.set_xlabel('Number of Sides')
    ax.set_ylabel('Frequency')
    ax.set_title('Number of Sides per Cell')


    plt.savefig(savedir + "/"+edges_name+"_" + plot_name+".png")
    plt.close()

def plot_summary_hist(cell_data, savedir, edges_name):
    """
    plot histograms for all of the continuous cell data

    Parameters:
    cell_data (pandas dataframe): cell based data
    savedir (string): location to save plot
    edges_name(string): filename of trace
    """
    fig, axes=plt.subplots(nrows=2, ncols=4, sharex=False, sharey=True, figsize=(12,6))
    hist=cell_data.loc[:, ~cell_data.columns.isin(['cell_id','cell_edge_count','cell_perimeter_nd', 'cell_area_nd'])].hist(grid=False, ax=axes)
    [x.title.set_size(14) for x in hist.ravel()]
    plt.suptitle('Summary Histograms', x=0.5, y=1.05, ha='center', fontsize='xx-large')
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=14)
    plt.savefig(savedir+'/'+edges_name+'_continuous_data_summary.png')
    plt.close()