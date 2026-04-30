import copy, math
from cProfile import label
from ctypes.wintypes import RGB
import numpy as np
import matplotlib.pyplot as plt
import math
import utils.lab_utils_multi as multilab

#variables crations and display
X_train, y_train = multilab.load_house_data()
print(f'x inputs = {X_train} y inputs = {y_train}')
X_features = ['size(sqft)','bedrooms','floors','age']

_, _, hist = multilab.run_gradient_descent(X_train, y_train, 10, alpha= 1e-7)
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")

multilab.plot_cost_i_w(X_train, y_train, hist)


plt.show()
