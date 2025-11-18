"""
visualise.py

Functions to plot data

Natasha Cowley 2025/13/02

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection, LineCollection


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
    width = np.pi / bins_number
    ax = plt.subplot(1, 1, 1, projection='polar')
    bars = ax.bar(bins[:bins_number], n, width=width, bottom=0, align='edge',color='blue', edgecolor='k')
    for bar in bars:
        bar.set_alpha(0.5)
        
    ax.set_xticks(bins)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.yaxis.grid(False)
    ax.set_xlabel('Frequency')
    ax.set_title(plot_name)
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

def plot_summary_hist(cell_data, plot_name, savedir, edges_name):
    """
    plot histograms for all of the continuous cell data

    Parameters:
    cell_data (pandas dataframe): cell based data
    savedir (string): location to save plot
    edges_name(string): filename of trace
    """
    df=cell_data.loc[:, ~cell_data.columns.isin(['cell_id','cell_edge_count','cell_perimeter_nd', 'cell_area_nd', 'time', 'cell_centre_x', 'cell_centre_y'])]
    #fig, axes=plt.subplots(nrows=2, ncols=int(np.ceil(df.shape[1]/2)), sharex=False, sharey=True, figsize=(12,6))
    hist=df.hist()
    [x.title.set_size(14) for x in hist.ravel()]
    plt.suptitle('Summary Histograms', x=0.5, y=1.05, ha='center', fontsize='xx-large')
    #plt.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout()
    plt.savefig(savedir+'/'+edges_name+plot_name+'_continuous_data_summary.png')
    plt.close()

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

def plot_cell_centres(cell_centres, color):
    for i in range(len(cell_centres)):
        plt.plot(cell_centres[i,0],cell_centres[i,1],marker ='o',markersize=2, c=color)
        
def plot_cell_id(cell_centres):
    for i in range(len(cell_centres)):
        plt.text(cell_centres[i][0], cell_centres[i][1], str(i),fontsize= 5,color='k', horizontalalignment='center', verticalalignment='center')

def plot_alignment_axis(cell_centres,alignment_axis,c='k'):
    for i in range(len(cell_centres)):
            plt.quiver(cell_centres[i,0],cell_centres[i,1],np.cos(alignment_axis)[i],np.sin(alignment_axis)[i],facecolor = c,scale=110,width=0.002,headwidth=0.0,headlength=0.0,headaxislength=0.0)
            plt.quiver(cell_centres[i,0],cell_centres[i,1],-np.cos(alignment_axis)[i],-np.sin(alignment_axis)[i],facecolor = c,scale=110,width=0.002,headwidth=0.0,headlength=0.0,headaxislength=0.0)

def plot_edges(A, R, c='k'):
    N_e=np.shape(A)[0]
    beg_edge = ((abs(A) - A)*0.5)@R
    end_edge = ((abs(A) + A)*0.5)@R
    for j in range(0,N_e):
        if (beg_edge[j,0]!=0) and (end_edge[j, 0]!=0):
            plt.plot([beg_edge[j,0],end_edge[j,0]],[beg_edge[j,1],end_edge[j,1]],c,alpha=1.0,linestyle ='-')


def plot_polys(C, R, cell_centres):
    N_c=np.shape(C)[0]
    patches = []

    for i in range(N_c):
        polygon = make_polygon(i, C, R, cell_centres)
        patches.append(polygon)

    p = PatchCollection(patches,alpha = 1.0)
    return p

def plot_binary_polys(C, R, cell_centres, plot_var, threshold):
    N_c=np.shape(C)[0]
    patches = []
    patchesLow = []
    patchesHigh = []

    for i in range(N_c):
        polygon = make_polygon(i, C, R, cell_centres)
        patches.append(polygon)
        
        if plot_var[i] < threshold:
            polygonLow = polygon
            patchesLow.append(polygonLow)
        else:
            polygonHigh = polygon
            patchesHigh.append(polygonHigh)


    ### For binary effective pressure
    p_low = PatchCollection(patchesLow,alpha = 0.5)
    p_high = PatchCollection(patchesHigh,alpha = 0.7)
   
    return p_low, p_high

def cell_plot_continuous(data,plot_variable, plot_label, cmap, cc_flag, cell_id_flag, shape_angle_flag,stress_angle_flag,C,R,cell_centres, edges_name, plot_dir):
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    polys=plot_polys(C, R, cell_centres)
    polys.set_array(data[plot_variable])
    polys.set_cmap(cmap) ###set polygon colourmap here
    polys.set_edgecolor('black') #black edges
    polys.set_clim(np.min(data[plot_variable]),np.max(data[plot_variable]))
    ax.add_collection(polys) 
    cbar = fig.colorbar(polys, ax=ax)
    cbar.ax.set_ylabel(plot_label, rotation=90) ###set colorbar label
    ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.set_xlim(min(R[:,0]-10), max(R[:,0]+10)) #if no edditional elements polygons won't show without manual setting of axes
    ax.set_ylim(min(R[:,1]-10), max(R[:,1]+10))

    
    if cc_flag==1:
        plot_cell_centres(cell_centres, 'black')
    if cell_id_flag==1:
        plot_cell_id(cell_centres)
    if shape_angle_flag==1:
        plot_alignment_axis(cell_centres,np.asarray(data['long_axis_angle']))
    if stress_angle_flag==1:
        plot_alignment_axis(cell_centres,np.asarray(data['stress_angle']))

    t=np.unique(data.time)[0]
    ax.set_title("t = {}s, {}".format(t,plot_label)) ###change title

    plt.tight_layout()
  
    plt.savefig(plot_dir+'/'+edges_name+'_'+plot_variable+'.png', dpi=300) ##edit filename here
    plt.close()

def cell_plot_discrete(data,plot_variable, plot_label, cmap, cc_flag, cell_id_flag, shape_angle_flag,stress_angle_flag,C,R,cell_centres, edges_name, plot_dir):
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    polys=plot_polys(C, R, cell_centres)
    polys.set_array(data[plot_variable])

    bounds = np.arange(data[plot_variable].min()-0.5, data[plot_variable].max()+1.5, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    polys.set_cmap(cmap) 
    polys.set_edgecolor('black') #black edges
    ax.add_collection(polys) 
    cbar=fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax, ticks=np.arange(data[plot_variable].min(), data[plot_variable].max()+1, 1))
    cbar.ax.set_ylabel(plot_label, rotation=90) ###set colorbar label
    ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.set_xlim(min(R[:,0]-10), max(R[:,0]+10)) #if no edditional elements polygons won't show without manual setting of axes
    ax.set_ylim(min(R[:,1]-10), max(R[:,1]+10))

    
    if cc_flag==1:
        plot_cell_centres(cell_centres, 'black')
    if cell_id_flag==1:
        plot_cell_id(cell_centres)
    if shape_angle_flag==1:
        plot_alignment_axis(cell_centres,np.asarray(data['long_axis_angle']))
    if stress_angle_flag==1:
        plot_alignment_axis(cell_centres,np.asarray(data['stress_angle']))

    t=np.unique(data.time)[0]
    ax.set_title("t = {}s, {}".format(t,plot_label)) ###change title

    plt.tight_layout()
  
    plt.savefig(plot_dir+'/'+edges_name+'_'+plot_variable+'.png', dpi=300) ##edit filename here
    plt.close()


def cell_plot_binary(data,plot_variable, plot_label,threshold, lowcolour, highcolour, cc_flag, cell_id_flag, shape_angle_flag,stress_angle_flag,C,R,cell_centres, edges_name, plot_dir):
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    polys_low, polys_high=plot_binary_polys(C, R, cell_centres,data[plot_variable], threshold)
    polys_low.set_facecolor(lowcolour) 
    polys_low.set_edgecolor('black') 
    ax.add_collection(polys_low)

    polys_high.set_facecolor(highcolour) 
    polys_high.set_edgecolor('black') 
    ax.add_collection(polys_high)

    ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.set_xlim(min(R[:,0]-10), max(R[:,0]+10)) #if no edditional elements polygons won't show without manual setting of axes
    ax.set_ylim(min(R[:,1]-10), max(R[:,1]+10))

    
    if cc_flag==1:
        plot_cell_centres(cell_centres, 'black')
    if cell_id_flag==1:
        plot_cell_id(cell_centres)
    if shape_angle_flag==1:
        plot_alignment_axis(cell_centres,np.asarray(data['long_axis_angle']))
    if stress_angle_flag==1:
        plot_alignment_axis(cell_centres,np.asarray(data['stress_angle']))

    t=np.unique(data.time)[0]
    ax.set_title("t = {}s, {}".format(t,plot_label)) ###change title

    plt.tight_layout()
  
    plt.savefig(plot_dir+'/'+edges_name+'_'+plot_variable+'_binary.png', dpi=300) ##edit filename here
    plt.close()


def cell_plot_ids(t,C,R,cell_centres, edges_name, plot_dir):
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    polys=plot_polys(C, R, cell_centres)
    polys.set_facecolor('white') ###set polygon colourmap here
    polys.set_edgecolor('black') #black edges
    ax.add_collection(polys) 
    ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.set_xlim(min(R[:,0]-10), max(R[:,0]+10)) #if no edditional elements polygons won't show without manual setting of axes
    ax.set_ylim(min(R[:,1]-10), max(R[:,1]+10))

    plot_cell_id(cell_centres)
    ax.set_title("t = {}s, {}".format(t,'cell IDs')) ###change title

    plt.tight_layout()
  
    plt.savefig(plot_dir+'/'+edges_name+'_'+'cell_ids'+'.png', dpi=300) ##edit filename here
    plt.close()

def edge_plot_continuous(data,plot_variable, plot_label, cmap, cc_flag, cell_id_flag, shape_angle_flag,stress_angle_flag,A,R,cell_centres, edges_name, plot_dir):
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    
    beg_edge = ((abs(A) - A)*0.5)@R
    end_edge = ((abs(A) + A)*0.5)@R
    edges=LineCollection(np.reshape(np.hstack((beg_edge, end_edge)), (len(A), 2,2)))
    edges.set_array(plot_variable)
    edges.set_cmap(cmap) ###set polygon colourmap here
    edges.set_clim(np.min(plot_variable),np.max(plot_variable))
    ax.add_collection(edges) 
    cbar = fig.colorbar(edges, ax=ax)
    cbar.ax.set_ylabel(plot_label, rotation=90) ###set colorbar label
    ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.set_xlim(min(R[:,0]-10), max(R[:,0]+10)) #if no edditional elements polygons won't show without manual setting of axes
    ax.set_ylim(min(R[:,1]-10), max(R[:,1]+10))

    
    if cc_flag==1:
        plot_cell_centres(cell_centres, 'black')
    if cell_id_flag==1:
        plot_cell_id(cell_centres)
    if shape_angle_flag==1:
        plot_alignment_axis(cell_centres,np.asarray(data['long_axis_angle']))
    if stress_angle_flag==1:
        plot_alignment_axis(cell_centres,np.asarray(data['stress_angle']))

    t=np.unique(data.time)[0]
    ax.set_title("t = {}s, {}".format(t,plot_label)) ###change title

    plt.tight_layout()
  
    plt.savefig(plot_dir+'/'+edges_name+'_'+plot_label+'_edges.png', dpi=300) ##edit filename here
    plt.close()
