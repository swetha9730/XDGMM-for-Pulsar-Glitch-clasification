#importing required libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
#from astroML.density_estimation import XDGMM
from scipy.stats import norm

from xdgmm import XDGMM

#importing data of the glitches 
data = pd.read_excel('data.ods', engine='odf',usecols = [0,1,2])
data = data.sort_values(by ='PSR name', ascending = 1)

#changing all the values to float and sting values to NaN
d1 = data['MJD'].apply(pd.to_numeric, errors='coerce')
data = data.drop(['MJD'], axis=1)
data = data.join(d1)

d2 = data['err'].apply(pd.to_numeric, errors='coerce')
data = data.drop(['err'], axis=1)
data = data.join(d2)

#print(data.info())
#dropping rows
data = data.dropna()
#print(data.info())
#print(data)

df = list(data.groupby(['PSR name']))
  
#print(len(df))

log_ti = []
err_log_ti = []

for i in range(len(df)):
    tmp = df[i][1]
    tmp = tmp.sort_values('MJD')
    tmp['ti']=tmp['MJD'].diff()

    for j in range(tmp.shape[0]):
        if j!=0:
            curr = tmp['err'].iloc[j]
            prev = tmp['err'].iloc[j-1]

            s = (curr**2 + prev**2)**0.5
            t = tmp['ti'].iloc[j]
            if t ==0.0:
                print(tmp.iloc[j])

            elti = s/((math.log(10))*(t))
            lti = math.log10(t)

            err_log_ti.append(elti)
            log_ti.append(lti)

#print(len(log_ti))
#print(len(err_log_ti))


#AIC/BIC
##COMPUTING XD
#stacking the data for computation
X = np.vstack([log_ti]).T
Xerr = np.zeros(X.shape + X.shape[-1:])
diag = np.arange(X.shape[-1])
Xerr[:, diag, diag] = np.vstack([err_log_ti]).T

# Instantiate an XDGMM model:
xdgmm = XDGMM()

"""
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
plt.savefig("AIC_BIC_interglitch.pdf", format="pdf") 
plt.show()

"""
# Looping over component numbers, fitting XDGMM model and computing the BIC and AIC and finding the optimal no of components :
optimal_n_comp = 2

xdgmm.n_components = optimal_n_comp
xdgmm = xdgmm.fit(X, Xerr)

#finding mean and std to plot the gaussians of the groups in data
mu1 = xdgmm.mu[0][0]
mu2 = xdgmm.mu[1][0]

c1 = xdgmm.V[0][0][0]
c2 = xdgmm.V[1][0][0]

w1 = xdgmm.weights[0]
w2 = xdgmm.weights[1]

std1 = np.sqrt(c1)
std2 = np.sqrt(c2)

#range of values over which data is spread
xmax = 4.015443587951102
xmin = 1.018284308426543

x = np.linspace(xmin, xmax, 100)

#defining the PDFs of the gaussians
p1 = norm.pdf(x, mu1, std1)*w1
p2 = norm.pdf(x, mu2, std2)*w2

#plotting
fig, ax = plt.subplots(1, 1, figsize = (10,7))

ax.hist((log_ti), density = True, bins = 20, histtype = 'step', color = 'lightblue', lw = 3)
ax.plot(x, p1, 'k', lw = 3)
ax.plot(x, p2, 'r', lw = 3)
ax.set_xlabel('Log($\Delta t_i$)', fontsize = 17,fontweight='bold')
ax.set_ylabel('Density of glitches', fontsize = 17,fontweight='bold')
ax.set_title('Extreme Deconvolution' ,fontsize = 18,fontweight='bold')
ax.tick_params(axis="x", labelsize=13) 
ax.tick_params(axis="y", labelsize=13) 

#print(mu1, std1,mu2,std2)

#ax.fill_between(a,b)

#olap = plt.fill_between(x[x>r], 0, norm.pdf(x[x>r],mu1,std1),alpha=0.3)
#olap = plt.fill_between(x[x<r], 0, norm.pdf(x[x<r],mu2,std2),alpha=0.3)
#plt.savefig("XD_interglitch.pdf", format="pdf") 
#plt.savefig("XD_interglitch_2comp.pdf", format="pdf") 
plt.show()
