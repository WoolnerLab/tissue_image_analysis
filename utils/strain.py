import numpy as np

from scipy import optimize, linalg

from utils import mechanics

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