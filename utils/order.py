"""
order.py

Functions to order vertices in cells

"""
import numpy as np

def order_verts(cell_id, cell_centre_a, cell_centre_b, R_a, R_b, C_a, C_b):
    #get vertex positions from ids
    cell_verts_a=R_a[np.where(C_a[cell_id]!=0)[0]]
    cell_verts_b=R_b[np.where(C_b[cell_id]!=0)[0]]

    
    Ralpha_a=cell_verts_a-cell_centre_a[cell_id] # vertex coordinates in frame of reference of cell
    Ralpha_b=cell_verts_b-cell_centre_b[cell_id] 
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