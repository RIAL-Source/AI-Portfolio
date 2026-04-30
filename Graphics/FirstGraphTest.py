from cProfile import label
from ctypes.wintypes import RGB
import numpy as np
import matplotlib.pyplot as plt
from utils.lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
plt.style.use('./deeplearning.mplstyle')

#Create a linear regression model for House Pricing

#variables crations and display
x_train = np.array([1.0, 2.0])
print(f'x inputs = {x_train}')
y_train = np.array([300.0, 500.0])
print(f'y inputs = {y_train}')
m = x_train.shape
print(f'Training examples are = {m}')



#
def compute_cost(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

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


temp_f_wb = np.array(compute_cost(x_train, 120, 200))

#creation of the graphic table
plt.title('Price of Houses')
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (in 1000s of dollars)')
plt.plot(x_train, temp_f_wb, c = 'b', label = 'Our Prediction')
plt.scatter(x_train,y_train, marker = 'x', c = 'r', label = 'Actual values')
#plt.axis([0,3500,0,1000])
plt.legend()
plt.show()

