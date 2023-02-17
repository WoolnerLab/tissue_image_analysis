"""
trace.py

code for running traces

Created by Natasha Cowley on 2023-01-24, adapted from code by Emma Johns and Alex Nestor-Bergmann
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from source import edge_detection
from source import setup_points

def run_trace(edges_name, nuclei_name,nuclei_exist, input_dir):
    #Find Nuclei

    if nuclei_exist==1:
        nuclei_file=input_dir+nuclei_name+'.tif'
        # Get the nuclei.
        im1, nuclei = setup_points.read_nuclei_image(nuclei_file, neighborhood_size=5)
    else: nuclei_file=None

    #Find Edges
    holes_file=None
    edges_file=input_dir+edges_name+'.tif'
    eptm = edge_detection.watershed_edges(nuclei_file, edges_file, holesFilename=holes_file, smoothing=1)

    eptm.update_geometry()
    eptm.update_neighbours()

    cellNumber = np.shape(eptm.cells)[0]
    print('Number of cells: '+str(cellNumber))

    cells = [c for c in eptm.cells]

    # Get unique vertices in tissue
    all_vertices = set([tuple(v) for c in cells for v in c.triJunctions])
    # Convert to list
    all_vertices = list(all_vertices)

    # Get unique edges in tissue
    all_edges = set([frozenset([tuple(e[0]), tuple(e[1])]) for c in cells for e in c.get_edges()])
    # Convert fozensets to tuples
    all_edges = [tuple(e) for e in all_edges]
    # Store the indices that reference the vertices
    all_edges_as_vertex_indices = [sorted([all_vertices.index(e[0]), all_vertices.index(e[1])]) for e in all_edges]

    # Save the list of unique vertices
    eptm.global_vertices = all_vertices

    # Save the list of unique edges that reference unique vertices
    eptm.global_edges = all_edges_as_vertex_indices

    edgeNumber = np.shape(eptm.global_edges)[0]
    print('Number of edges: '+str(edgeNumber))
    vertexNumber = np.shape(eptm.global_vertices)[0]
    print('Number of vertices: '+str(vertexNumber))

    # Loop over all cells to store the above data locally per cell (cell centroids not edge centroids)
    centroidsX, centroidsY = [], []

    for c in cells:
        centroid = c.get_centroid()
        centroidsX.append( centroid[0] )
        centroidsY.append( centroid[1] )


        # Save the indices of the unique vertices it has, referencing eptm.global_vertices list
        c.global_linked_vertex_ids = [all_vertices.index(v) for v in c.triJunctions]

        # For edges, first store a list of edges referencing the indices of the unique vertices
        c.global_linked_edges = [sorted([all_vertices.index(tuple(e[0])), all_vertices.index(tuple(e[1]))])
                                    for e in c.get_edges()]
        # Now store a list of edges that references the list of unique edges (eptm.global_edges)
        c.global_linked_edge_ids = [all_edges_as_vertex_indices.index(e) for e in c.global_linked_edges]
        #print(c.global_linked_edge_ids)

    # Get cell pairs that reference edges (index is edge number, and each entry are the cells connected to that edge)
    eptm.cells_connected_to_edges = [[cell_id for cell_id in range(len(cells))
                                        if edge_id in cells[cell_id].global_linked_edge_ids]
                                        for edge_id in range(len(eptm.global_edges))]
    #print(eptm.cells_connected_to_edges)
    edgeToCell = eptm.cells_connected_to_edges


    #make a dictionary of edges with cell ID as key.
    cellEdges = {}

    for i in range(0,edgeNumber): #loop over list of edges containing connected cells
        oneOrTwo = np.shape(edgeToCell[i])[0]
        j = edgeToCell[i][0] #for first element (eg first cell)
        if j in cellEdges.keys(): #if already in dictionary append edge number as a value
            cellEdges[j].append(i) 
        else: #if not then add then add a new dictionary entry for the cell  j with value i
            cellEdges[j] = [i]
        if oneOrTwo>1: #if edge is connecting 2 cells
            j = edgeToCell[i][1] #2nd cell
            if j in cellEdges.keys():
                cellEdges[j].append(i)
            else:
                cellEdges[j] = [i]




    return edgeToCell, centroidsX, centroidsY,  eptm, edgeNumber, cellEdges


def check_trace_plot(savedir, eptm):
    """
    Plots the network configuration to allow for visual checking.
    """
    f, ax = plt.subplots()
    for c in eptm.cells:
        verts = [eptm.global_vertices[v] for v in c.global_linked_vertex_ids]
        x, y = zip(*verts)
        ax.plot(x, y, 'o')

        edges = [(eptm.global_vertices[e[0]], eptm.global_vertices[e[1]]) for e in c.global_linked_edges]
        edges1 = matplotlib.collections.LineCollection(edges, linewidth=1, color='black', alpha=0.5)
        ax.add_collection(edges1)

    plt.gca().set_aspect('equal')
    plt.title('Image in Pixels to check')

    plt.savefig(savedir +'/checkImage.png')