"""
spatial.py
Natasha Cowley 2024/07/16

Functions to calculate spatial quatities of cells network

"""

import numpy as np

from scipy import optimize, linalg

from utils import mechanics


def get_tangents(A,R):
    """
    calculate tangent edges, shape (Ne,2)
    """
    return A@R  #tangents, t_j=A_jk r_k

def get_edge_lengths(tangents):
    """
    calculate length of edges
    """
    return np.sqrt(np.sum(tangents*tangents, axis=1)) # edge length |t_j|

def get_edge_centroids(A,R):
    """
    calculate edge centroids
    """
    return 0.5*abs(A)@R #centroids, c_j=1/2 abs(A)_jk r_k

def get_boundary_vertices(A, B, N_c):
    """
    find boundary vertices
    """
    return 0.5*(abs(A).T@(B.T@np.ones(N_c))**2) #vertices on the periphery

def get_edge_count(B):
    """
    count the number of edges per cell
    """
    return  np.sum(abs(B),axis=1) #edges per cell

def get_cell_centres(C, R, cell_edge_count):
    """
    calculate cell centre positions
    """
    return ((C@R).T/cell_edge_count).T #cell centres, R_i

def get_peripheral_edge_centroids(B, edge_centroids, N_c):
    """
    find edge centroids on the network periphery
    """
    return(abs(np.ones(N_c)@B)*edge_centroids.T).T
    
def get_normals(B, tangents, N_c, N_e):
    """
    calculate outward normals for edge j in cell i
    """
    epsilon_c=np.array([[0,1], [-1,0]]) # 2D Levi-Cevita tensor
    nij=np.zeros((N_c,N_e, 2) ) #outward normal to edge
    for i in range(N_c):
        for j in range(N_e):
            nij[i,j]=-epsilon_c@(B[i,j]*tangents[j])
    return nij

def get_perimeters(A,B,R):
    """
    calculate cell perimeters
    """
    tangents=get_tangents(A,R)
    l=get_edge_lengths(tangents)
    L= abs(B)@l #perimeters, L_i=abs(B)ij |t_j|
    return L


def get_areas_old(A, B, R):
    """
    calculate cell areas Legacy function in terms of matrices
    """
    N_c=np.shape(B)[0]
    N_e=np.shape(B)[1]

    tangents=get_tangents(A,R)
    c=get_edge_centroids(A,R)
    nij=get_normals(B, tangents, N_c, N_e)

    cell_areas=np.zeros(N_c) #cell area
    for i in range(N_c):
        for j in range(N_e):
            cell_areas[i]+=0.5*np.dot(nij[i,j],c[j])
    return abs(cell_areas)

def get_areas(cells, R):
    """
    calculate cell areas
    """
    areas=np.array([shapely.Polygon(R[cells[x]]).area for x in range(len(cells))])
    return areas

def get_mean_area(cell_areas):
    """
    calculate the mean area of cells and also write cell areas to file.
    """

    mean_cell_area = np.mean(cell_areas)
    if mean_cell_area <0: mean_cell_area*=-1
    np.savetxt('cell_areas.txt',cell_areas)
    return mean_cell_area

def get_pref_area(cell_areas, gamma, L, L_0, mean_cell_area):
    """
    Uses Newtons method to find the value of the dimensional area when the global stress is zero.
    """
    sol = optimize.root_scalar(mechanics.GlobalStress,args=(cell_areas, gamma, L, L_0),  x0=mean_cell_area, fprime=mechanics.derivative_GlobalStress, method='newton')
    pref_area=sol.root
    return pref_area


def get_shape_tensor(R,C,cell_edge_count,cell_centres,cell_P_eff=False):
    """
    celculate the shape tensor and get principal axis and circularity
    """
    N_c=np.shape(C)[0]
    N_v=np.shape(C)[1]

    circularity =np.zeros((N_c))
    eigen_val_store = np.zeros((N_c,2))
    eigen_vector_store1 = np.zeros((N_c,2))
    eigen_vector_store2 = np.zeros((N_c,2))
    major_shape_axis_store = np.zeros((N_c,2))
    major_shape_axis_alignment = np.zeros((N_c))
    major_stress_axis_store = np.zeros((N_c,2))
    major_stress_axis_alignment = np.zeros((N_c))

    for i in range(N_c):
        shape_tensor = np.zeros((2,2))
        for k in range(N_v):
            if C[i,k]==1.0:
                shape_tensor += (1.0/cell_edge_count[i])*np.outer((R[k]-cell_centres[i]),(R[k]-cell_centres[i]))

        eigvals, eigvecs = linalg.eig(shape_tensor)
        eigvals = eigvals.real
        if eigvals[0] > eigvals[1]:
            eigen_val_store[i,0]= eigvals[0]
            eigen_val_store[i,1]= eigvals[1]
            eigen_vector_store1[i] = eigvecs[:,0]
            eigen_vector_store2[i] = eigvecs[:,1]

        else:
            eigen_val_store[i,0]= eigvals[1]
            eigen_val_store[i,1]= eigvals[0]
            eigen_vector_store2[i] = eigvecs[:,0]
            eigen_vector_store1[i] = eigvecs[:,1]

        major_shape_axis_store[i] = eigen_vector_store1[i]

        
        major_shape_axis_alignment[i] = np.arctan2(major_shape_axis_store[i][1]/(np.sqrt(major_shape_axis_store[i][0]**2+major_shape_axis_store[i][1]**2)),major_shape_axis_store[i][0]/(np.sqrt(major_shape_axis_store[i][0]**2+major_shape_axis_store[i][1]**2)))
        if major_shape_axis_alignment[i]<0:
            major_shape_axis_alignment[i] +=np.pi

        circularity[i] = abs(eigen_val_store[i][1]/eigen_val_store[i][0])


        if len(cell_P_eff)>1:
            if cell_P_eff[i]<0:
                major_stress_axis_store[i] = eigen_vector_store2[i]
            else:
                major_stress_axis_store[i] = eigen_vector_store1[i]
            
            major_stress_axis_alignment[i] = np.arctan2(major_stress_axis_store[i][1]/(np.sqrt(major_stress_axis_store[i][0]**2+major_stress_axis_store[i][1]**2)),major_stress_axis_store[i][0]/(np.sqrt(major_stress_axis_store[i][0]**2+major_stress_axis_store[i][1]**2)))
            if major_stress_axis_alignment[i]<0:
                major_stress_axis_alignment[i] +=np.pi
    if len(cell_P_eff)>1:
        return circularity,major_shape_axis_store,major_shape_axis_alignment,major_stress_axis_store,major_stress_axis_alignment
    else:
        return circularity,major_shape_axis_store,major_shape_axis_alignment
    
def get_nearest_neighbours(B, n):
    nn=np.unique(np.where(B[:,np.where(B[n]!=0)[0]]!=0)[0])
    nn=nn[nn!=n]
    return(nn)

def get_all_nn(B):
    all_nn=[]
    for i in range(len(B)):
        nn=get_nearest_neighbours(B,i)
        all_nn.append(nn)
    return (all_nn)

def get_dAdr(A, nij):
    dAdr=0.5*np.transpose(np.tensordot(abs(A).T, np.transpose(nij, (1, 0, 2)), axes=1), (1,0,2))

    return dAdr

def get_dLdr(A, B, tangents, edge_lengths):
    N_e=len(edge_lengths)
    diagthat=np.zeros((N_e, N_e, 2))
    that=(tangents/edge_lengths[:,None])
    for j in range(N_e):
        diagthat[j,j]=that[j] 
    dLdr=np.tensordot(abs(B),np.einsum('ijk, jl -> ilk',diagthat, A, optimize='optimal'), axes=1)
    return dLdr