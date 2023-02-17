"""
matrices.py

Functions to calculate incidence matrices, adapted from code by Emma Johns

"""
import numpy as np


def get_matrices(edges, vertices, cells, cX, cY):
    """
    Calculates the matrices A (incidence matrix of edges and vertices, Ne x Nv), 
    B (incidence matrix of cell faces and edges, Nc x Ne) and C, (relates faces and vertices, Nc x Nv)

    Parameters:
        edges (numpy array): array of edges with vertex indexes
        vertices (numpy array): array of vertex coordinates
        cells (dictionary): keys: cell ids, values: edge ids
        cX (numpy array): cell centre x coordinate
        cY (numpy array): cell centre y coordinate

    Returns:
        A (numpy array): Ne x Nv order array relating edges to vertices
        B (numpy array): Nc x Ne order array relating cells to edges
        C (numpy array): Nc x Nv order array relating cells to vertices
        R (numpy array): vertex positions
    """
    N_c = len(cX)
    N_e = len(edges)
    N_v = len(vertices)

    cell_centre=np.transpose(np.asarray([cX,cY]))

    R=np.asarray(vertices, dtype=float)

    A = np.zeros((N_e, N_v))
    max_v=np.max(edges, axis=1)
    min_v=np.min(edges, axis=1)

    for j in range(0,N_e):
        A[j][max_v[j]]=1 #flows into vertex
        A[j][min_v[j]]=-1 #flows out of vertex
        
    B = np.zeros((N_c,N_e))


    for i in range(0,N_c): #loop over cells
        edge_cell = cells[i] #get edge ids connected to each cell (cells is a dictionary)
        no_sides = len(edge_cell) 
        edge_cell=np.array(edge_cell).astype(int)
        edge_vert_id=edges[edge_cell]


        #get vertex positions from ids
        coord_verts=R[np.unique(edge_vert_id)]
        
        Ralpha=coord_verts-cell_centre[i] # vertex coordinates in frame of reference of cell
        ang=np.arctan2(Ralpha[:,1], Ralpha[:,0])%(2*np.pi) #angle wrt to +ve horizontal x-axis
        
        gs=np.transpose(np.vstack((np.unique(edge_vert_id), ang))) #get unique edges and combine with angle
        gs=gs[np.argsort(-gs[:,1], axis=0)] #order vertex ids by clockwise angle from theta=0

        max_ev = np.max(edge_vert_id, axis=1)
        min_ev = np.min(edge_vert_id, axis=1)
        
        #for each edge find the location of the min and max vertex if the min vertex is 1 before the 
        # max vertex in the ordered list then the edge is going clockwise round the cell,
        #if not then it is going anti clockwise
        for j in range(0,no_sides):
            
            a=np.where(gs[:,0].astype(int)==max_ev[j])[0][0]
            b=np.where(gs[:,0].astype(int)==min_ev[j])[0][0]      
            
            if a-b==1 or a-b == -(no_sides-1):
                B[i][int(edge_cell[j])] = 1 #clockwise
            else:
                B[i][int(edge_cell[j])] = -1 #anticlockwise


    #Check the matrix multiplication of B and A, BA=0, if not there is an issue
    if np.where((B@A).flatten()!=0)[0].size!=0: raise Exception('BA!=0.Incorrect matrix calculation')

    #calculate cell vertex matrix C_ik
    C=0.5*np.matmul(abs(B), abs(A))

    return A, B, C, R