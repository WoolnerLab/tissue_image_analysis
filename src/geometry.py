"""
geometry.py
Natasha Cowley 2024/07/16

Functions to calculate spatial quatities of cells network

"""

import numpy as np
import shapely 

from scipy import optimize, linalg

from src import mechanics


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
    calculate cell areas (Legacy function in terms of matrices)
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
    return mean_cell_area

def get_pref_area(cell_areas, gamma, L, L_0, mean_cell_area):
    """
    Uses Newtons method to find the value of the dimensional area when the global stress is zero.
    """
    sol = optimize.root_scalar(mechanics.GlobalStress,args=(cell_areas, gamma, L, L_0),  x0=mean_cell_area, fprime=mechanics.derivative_GlobalStress, method='newton')
    pref_area=sol.root
    return pref_area

def get_shape_tensors(R,cells, cell_edge_count, cell_centres):
    """
    calculate the shape tensor
    """

    N_c=len(cells)
    N_v=len(R)
    R_alpha=[R[cells[x]] - cell_centres[x] for x in range(N_c)] #R ref frame of cell
    S=np.array([(R_alpha[i].T@R_alpha[i])/cell_edge_count[i] for i in range(N_c)])

    return S

def get_circularity(S):
    "calculate circularity"
    evals=np.sort(np.linalg.eigvals(S), axis=1)
    circ=np.abs(evals[:,0]/evals[:,1])
    return circ, evals
    

def get_shape_axis_angle(S):
    N_c=len(S)
    evals, evecs=np.linalg.eig(S)
    long_axis=np.array([evecs[x][:,evals[x].argmax()] for x in range(N_c)])

    long_axis_angle = np.arctan2(long_axis[:,1],long_axis[:,0])
    long_axis_angle=np.where(long_axis_angle<0, long_axis_angle+np.pi, long_axis_angle)

    return long_axis_angle

def get_edge_angle(tangent):
    edge_angle = np.arctan2(tangent[:,1],tangent[:,0])
    edge_angle=np.where(edge_angle<0, edge_angle+np.pi, edge_angle)

    return edge_angle
    
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

def get_nn_shells(B):
    """
    Get matrix of shells of nearest neighbours
    """

    N_c=np.shape(B)[0]

    all_nn=[]
    for i in range(N_c):
        nn=[]
        all_cells=[]
        nn.append([i])
        all_cells.append(i)

        shell1=list(np.unique(np.where(B[:,np.where(B[i,:]!=0)[0]]!=0)[0])) #1st shell of neighbouring cells
        shell1.remove(i)
        
        nn.append(shell1)
        all_cells.extend(shell1)

        while len(all_cells) < N_c: 
            shell=[]
            for n in nn[-1]: #for each cell in previous shell find their nns not in previous cell
                nn_i=np.unique(np.where(B[:,np.where(B[n,:]!=0)[0]]!=0)[0])
                shell.append(set(nn_i).difference(all_cells))
            shell=set([x for xs in shell for x in xs]) #get unique cells
            nn.append(list(shell))
            all_cells.extend(list(shell))
        
        all_nn.append(nn)
        
    shell_matrix=np.array([[[x for x, xs in enumerate(all_nn[m]) if n in xs][0] for n in range(N_c)] for m in range(N_c)])


    return shell_matrix

def get_Q_J(tangents, edge_lengths, B,cell_perimeters):
    N_c=np.shape(B)[0] #number of cells
    N_e=np.shape(B)[1] #number of edges

    Q=np.zeros((N_c,2,2))
    J=np.zeros((N_c,2,2))
    for i in range(N_c):
        for j in range(N_e):
            if abs(B[i,j])==1:#makes sure we are in the cell i
                that=(tangents[j]/edge_lengths[j])
                Q[i]+=abs(B)[i,j]*edge_lengths[j]*np.outer(that,that)
        Q[i]=Q[i]/(cell_perimeters[i])

        J[i]=Q[i]-0.5*np.identity(2)
    
    return Q,J