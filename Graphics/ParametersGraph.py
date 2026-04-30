from cProfile import label
from ctypes.wintypes import RGB
import numpy as np
import matplotlib.pyplot as plt
from utils.lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
plt.style.use('./deeplearning.mplstyle')


#variables crations and display
x_train = np.array([1.0, 2.0])
print(f'x inputs = {x_train}')
y_train = np.array([300.0, 500.0])
print(f'y inputs = {y_train}')
m = x_train.shape
print(f'Training examples are = {m}')



def compute_cost(x,y,w,b):

    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    m_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i

    dj_db = dj_db / m
    dj_dw /= m

    return dj_dw, dj_db


plt_gradients(x_train,y_train, compute_cost, compute_gradient)
plt.show()




