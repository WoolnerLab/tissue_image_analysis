"""
mechanics.py

Functions to calculate machanical properties of the network
"""

import numpy as np

def get_P_eff(A, Gamma, L, L_0):
    """Calculate Effective pressure of cell"""
    return (A - 1) + Gamma*(L-L_0)*L/(2*A)

def GlobalStress(pref_area, A, Gamma, L, L_0):
    """GlobalsStress of the cells as a function of cell area"""
    globalStress = np.sum(((A/pref_area) - 1 + 0.5*Gamma*(L/A)*(L - L_0*(pref_area**(1/2))))*A)/np.sum(A);
    return globalStress;

def derivative_GlobalStress(pref_area, A, Gamma, L, L_0):
    """first derivative of global stress of the cells as a function of cell area"""
    d_globalStress = np.sum((-(A/(pref_area**2))- 0.25*Gamma*(L/A)*(L_0*(pref_area**(-1/2))))*A)/np.sum(A);
    return d_globalStress;

def get_cell_pressures(A):
    """ calculates non-dimensional cell pressures"""
    return A -1.0

def get_cell_tensions(Gamma, L, L_0):
    """ calculates non-dimensional cell tensions"""
    return Gamma*(L-L_0)

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

def calc_shear(tangents, edge_lengths,B,cell_perimeters,cell_tensions,cell_areas): #Look at this function
    """
    calculate shear stress and zeta matrix
    """
    # N_c=np.shape(B)[0] #number of cells
    # N_e=np.shape(B)[1] #number of edges


    # J=np.zeros((N_c,2,2))
    # for i in range(N_c):
    #     Q = np.zeros((2,2))
    #     for j in range(N_e):
    #         if abs(B[i,j])==1:#makes sure we are in the cell i
    #             that=(tangents[j]/edge_lengths[j])
    #             Q+=abs(B)[i,j]*edge_lengths[j]*np.outer(that,that)
    #     Q=Q/(cell_perimeters[i])

    #     J[i]=Q-0.5*np.identity(2)
    Q,J=get_Q_J(tangents, edge_lengths, B,cell_perimeters)


    detJ = np.linalg.det(J)

    cell_zetas = np.sqrt(-detJ)
    cell_shears = ((cell_perimeters*cell_tensions)/cell_areas)*cell_zetas

    return cell_shears,cell_zetas

def get_global_stress(P_eff, A):
    """ 
    Calculate global stress for he monolayer
    """
    return np.sum(P_eff*A)/np.sum(A)

def get_monolayer_energy(A, L, L_0):
    """
    Calculate total energy for the monolayer
    """
    return 0.5*np.sum((A-1)**2+(L-L_0)**2)