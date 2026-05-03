import numpy as np
import matplotlib.pyplot as plt
from utils.lab_utils_multi import plt_equal_scale
from normalization import *
from getdata import *
from HousePricing import *

np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

#set alpha to 1e-7
m,n = X_train.shape
# initialize parameters
initial_w = np.zeros(n)
initial_b = 0
_,_,hist = gradient_descent(X_train ,y_train, initial_w , initial_b, 1e-7, 
                                                    10, compute_cost, compute_gradient)
# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")


############## PREDICTIONS ##############

w_norm, b_norm, hist = gradient_descent(X_norm ,y_train, initial_w , initial_b, 1e-1, 
                                                    1000, compute_cost, compute_gradient)
#predict target using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features    
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp,color='#FF9300', label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")

plt_equal_scale(X_train, X_norm, y_train)

plt.show()

