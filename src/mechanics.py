"""
mechanics.py
Natasha Cowley 2024/07/16

Functions to calculate mechanical properties of cells network

"""

import numpy as np
from src import geometry

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



def calc_shear(tangents, edge_lengths,B,cell_perimeters,cell_tensions,cell_areas): #Look at this function
    """
    calculate shear stress and zeta matrix
    """

    Q,J=geometry.get_Q_J(tangents, edge_lengths, B,cell_perimeters)


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

def get_stress_angle(P_eff, shape_angle):
    #if P_eff < 0 get perpendicular angle
    stress_angle=np.where(P_eff<0, shape_angle-np.pi/2, shape_angle)
    stress_angle=np.where(stress_angle<0, stress_angle+np.pi, stress_angle)
    return stress_angle