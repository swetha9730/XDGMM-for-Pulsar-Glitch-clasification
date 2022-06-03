#importing required libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from astroML.density_estimation import XDGMM
from scipy.stats import norm

from xdgmm import XDGMM

#importing data of the glitches where 1st column is dV/V while 2nd column is the error in dV/V in the order of 10^(-9)
data = pd.read_excel('glitches.ods', engine='odf',header=None)
data.columns = ['derv','err_derv']

#changing all the values to float and sting values to NaN
data = data.apply(pd.to_numeric, errors='coerce')

#dropping rows
indexNames1 = data[(data['derv'] <= 0)].index
data.drop(indexNames1 , inplace=True) 
data = data.dropna()

#reallocating the arrays
derv = data['derv']
err_derv = data['err_derv']

#defining log and error
err_log_derv = (err_derv/((math.log(10))*(derv)))

log_derv = []
for item in derv:
    x = item
    y = (math.log10(x))-9
    log_derv.append(y)

data['log_derv'] = log_derv
data['err_log_derv'] = err_log_derv

log_derv = data['log_derv']
err_log_derv = data['err_log_derv']

##COMPUTING XD
#stacking the data for computation
X = np.vstack([log_derv]).T
Xerr = np.zeros(X.shape + X.shape[-1:])
diag = np.arange(X.shape[-1])
Xerr[:, diag, diag] = np.vstack([err_log_derv**2]).T

# Instantiate an XDGMM model:
xdgmm = XDGMM()

# Looping over component numbers, fitting XDGMM model and computing the BIC and AIC and finding the optimal no of components :
optimal_n_comp = 2

xdgmm.n_components = optimal_n_comp
xdgmm = xdgmm.fit(X, Xerr)

#finding mean and std to plot the gaussians of the groups in data
mu1 = xdgmm.mu[0][0]
mu2 = xdgmm.mu[1][0]

c1 = xdgmm.V[0][0]
c2 = xdgmm.V[1][0]

w1 = xdgmm.weights[0]
w2 = xdgmm.weights[1]

std1 = np.sqrt(c1)
std2 = np.sqrt(c2)

#range of values over which data is spread
xmax = -4.187086643357144
xmin = -11.602059991327963

x = np.linspace(xmin, xmax, 100)

#defining the PDFs of the gaussians
p1 = norm.pdf(x, mu1, std1)*w1
p2 = norm.pdf(x, mu2, std2)*w2

#plotting
fig, ax = plt.subplots(1, 1, figsize = (10,7))

ax.hist((log_derv), density = True, bins = 20, histtype = 'step', color = 'darkgrey', lw = 3)
ax.plot(x, p1, 'k', lw = 3)
ax.plot(x, p2, 'r', lw = 3)
ax.set_xlabel('Log(Δv/v)', fontsize = 17,fontweight='bold')
ax.set_ylabel('Density of glitches', fontsize = 17,fontweight='bold')
#ax.set_title('Extreme Deconvolution' ,fontsize = 18,fontweight='bold')
ax.tick_params(axis="x", labelsize=13) 
ax.tick_params(axis="y", labelsize=13) 
#ax.grid()
#plt.savefig("XD_fit.pdf", format="pdf") 

plt.show()
