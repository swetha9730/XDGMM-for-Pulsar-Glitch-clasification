#importing required libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from astroML.density_estimation import XDGMM
from scipy.stats import norm

from xdgmm import XDGMM

#importing data of the glitches where 1st column is dV/V while 2nd column is the error in dV/V in the order of 10^(-9)
data = pd.read_excel('data.ods', engine='odf',usecols = [3,4])
data.columns = ['derv','err_derv']

#changing all the values to float and sting values to NaN
data = data.apply(pd.to_numeric, errors='coerce')

#dropping rows
indexNames1 = data[(data['derv'] <= 0)].index
data.drop(indexNames1 , inplace=True) 

indexNames2 = data[(data['err_derv'] == 0)].index
data.drop(indexNames2 , inplace=True) 

data = data.dropna()
print(data.info())

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

# Defining the range of component numbers
param_range = np.array([1,2,3,4])

# Looping over component numbers, fitting XDGMM model and computing the BIC and AIC:
bic, optimal_n_comp, lowest_bic = xdgmm.bic_test(X, Xerr, param_range)
aic, optimal_n_comp1,lowest_aic = xdgmm.aic_test(X, Xerr, param_range)

#plotting
fig, ax = plt.subplots(1, 1, figsize = (10,7))

ax.plot(param_range, aic, 'o-', lw = 3, c = 'blue', label = 'AIC')
ax.plot(param_range, bic, '^:', lw = 3, c = 'red', label = 'BIC')
ax.legend(frameon=False, fontsize = 15)

ax.set_xlabel('Number of components', fontsize = 17,fontweight='bold')
ax.set_ylabel('Information criterion', fontsize = 17,fontweight='bold')

new_list = range(math.floor(min(param_range)), math.ceil(max(param_range))+1)
plt.xticks(new_list)
ax.tick_params(axis="x", labelsize=13) 
ax.tick_params(axis="y", labelsize=13) 

plt.savefig("AIC_BIC_updated.pdf", format="pdf") 
plt.show()
