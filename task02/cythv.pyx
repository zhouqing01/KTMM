from libc.math cimport sqrt
import numpy as np
cimport numpy as np

ctypedef np.double_t DTYPE_t

cdef double G = 6.6742 * 1e-11
def A_th(np.ndarray[DTYPE_t] pos_x, np.ndarray[DTYPE_t] pos_y, np.ndarray[DTYPE_t] mas, int pl_i, int size):
    cdef double a_x = 0
    cdef double a_y = 0
    cdef int j
    
    for j in range(size):
        if j != pl_i:
            if pos_x[j] != pos_x[pl_i]:
                a_x += G * mas[j] * (pos_x[j] - pos_x[pl_i]) / (sqrt( (pos_x[j] - pos_x[pl_i])**2 + (pos_y[j] - pos_y[pl_i])**2 ))**3
            
            if pos_y[j] != pos_y[pl_i]:
                a_y += G * mas[j] * (pos_y[j] - pos_y[pl_i]) / (sqrt( (pos_x[j] - pos_x[pl_i])**2 + (pos_y[j] - pos_y[pl_i])**2 ))**3
    return [a_x, a_y, pl_i]

def Pos(np.ndarray[DTYPE_t] pos_x, np.ndarray[DTYPE_t] pos_y, np.ndarray[DTYPE_t] v_x, np.ndarray[DTYPE_t] v_y, np.ndarray[DTYPE_t] a_x, np.ndarray[DTYPE_t] a_y, np.ndarray[DTYPE_t] mas, double delt_t, int size):
    cdef np.ndarray[DTYPE_t] out_x = np.zeros(size, dtype = np.double)
    cdef np.ndarray[DTYPE_t] out_y = np.zeros(size, dtype = np.double)

    for i in range(size):
        a = A_th(pos_x, pos_y, mas, i, size)
        a_x[i] = a[0]
        a_y[i] = a[1]
        out_x[i] = pos_x[i] + v_x[i] * delt_t + 0.5 * a_x[i] * delt_t**2
        out_y[i] = pos_y[i] + v_y[i] * delt_t + 0.5 * a_y[i] * delt_t**2
    
    return out_x, out_y, a_x, a_y

def Speed(np.ndarray[DTYPE_t] pos_x, np.ndarray[DTYPE_t] pos_y, np.ndarray[DTYPE_t] v_x, np.ndarray[DTYPE_t] v_y, np.ndarray[DTYPE_t] a_x, np.ndarray[DTYPE_t] a_y, np.ndarray[DTYPE_t] mas, double delt_t, int size):
    cdef np.ndarray[DTYPE_t] out_x = np.zeros(size, dtype = np.double)
    cdef np.ndarray[DTYPE_t] out_y = np.zeros(size, dtype = np.double)

    cdef np.ndarray[DTYPE_t] a_x_next = np.zeros(size, dtype = np.double)
    cdef np.ndarray[DTYPE_t] a_y_next = np.zeros(size, dtype = np.double)

    for i in range(size):
        a = A_th(pos_x, pos_y, mas, i, size)
        a_x_next[i] = a[0]
        a_y_next[i] = a[1]
        out_x[i] = v_x[i] + 0.5 * (a_x_next[i] + a_x[i]) * delt_t
        out_y[i] = v_y[i] + 0.5 * (a_y_next[i] + a_y[i]) * delt_t
    
    return out_x, out_y, a_x_next, a_y_next