#!/usr/bin/python

#THIS CODE GENERATES SEVERAL HISTOGRAMS(LINES)
#TO COMPARE ENERGY DISTRIBUTIONS
#DELIMITING ALSO THE SPNODAL REGIONS
#########LIBS############################3
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import os
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from cycler import cycler
###########################################################
fig, ax = plt.subplots()

########VARIABLES##############
N=50      #NUMBER OF BINS
Ndata=501 #NUMBER OF DATAFILES
Nstep=5   #SKIP BETWEEN DATAFILS TO BE PLOTTED
tempt=1   #TEMPERATURE
Tskip=0   #TIME INTERVAL TO BE SKIPPED
density=np.zeros(2*(Ndata-1))
##############################
k=1
for i in range(1,Ndata,Nstep):
    val ='%.3f'%((float(i)/500))
    density[i]=float(val)
    sufix=str('cluster_sys-D'+str(density[i])+'-T'+str(tempt)+'.data');
    if(os.path.isfile('./'+str(sufix))):
        k+=1
        Maxdata=(float(i)/500)
        Maxi=i
print("Couldn't find "+str(Ndata-k)+" files")
#Ndata=k*Nstep
#####COLORBAR###############
print(Maxdata)
x = range(Ndata)
scales = np.linspace(0, Maxdata, Ndata)
locs = range(4)
cmap = plt.get_cmap("viridis")#SET THE COLORMAP
norm = plt.Normalize(scales.min(), scales.max())
fig, axes = plt.subplots(1,1, constrained_layout=True, sharey=True)
sm =  ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm)#
cbar.ax.set_title(r'Density$\left(\rho*\right)$') # = \rho \sigma^{dim}
#########################
colors = cmap(np.linspace(0,1,Maxi+1))
for i in range(1,Ndata,Nstep):
    val ='%.3f'%((float(i)/500))
    density[i]=float(val)
    sufix=str('cluster_sys-D'+str(density[i])+'-T'+str(tempt)+'.data');
    if(os.path.isfile('./'+str(sufix))):
        filename = str(sufix);
        data = pd.read_csv(filename, usecols = [0,1,2,3,4], names = ['T','N','KE','PE','Temp'] , header = None, skiprows = 1, delimiter=' ', skipinitialspace = True, skip_blank_lines=False) #READ THE FILE AND TRANSFER DATA TO A DATA FRAME
        a=data[data['T']>Tskip]

        x = np.array(a['PE'])   #turn x,y data into numpy arrays
        kwargs = dict(histtype='step', lw=2, alpha=1,density=True,bins=50,color=colors[i],label="_nolegend_") #
        z,w,_=plt.hist(x, **kwargs)
    else:
        print('file:'+str(sufix)+' doesnt exist, skipping')

data2 = pd.read_csv('Spinodal_Cv_L.csv', usecols = [0,1], names = ['rho','T'] , header = None, delimiter=' ', skipinitialspace = True, skip_blank_lines=True,skiprows = 1) #READ THE FILE AND TRANSFER DATA TO DATAFRAME
dots=data2[data2['T']==tempt]

val ='%.3f'%(float(dots['rho']))
sufix=str('cluster_sys-D'+str(val)+'-T'+str(tempt)+'.data');
filename = str(sufix);
print(filename)
data = pd.read_csv(filename, usecols = [0,1,2,3,4], names = ['T','N','KE','PE','Temp'] , header = None, skiprows = 1, delimiter=' ', skipinitialspace = True, skip_blank_lines=False) #READ THE FILE AND TRANSFER DATA TO A DATA FRAME
a=data[data['T']>Tskip]
x = np.array(a['PE'])
kwargs = dict(histtype='step',density=True,bins=50,alpha=1,lw=2,color='red',label=r'$\rho$='+str(val)) #
z,w,_=plt.hist(x, **kwargs)
print(float(dots['rho']))
cbar.ax.plot([0, 1], [float(dots['rho'])/Maxdata]*2, 'red')

data2 = pd.read_csv('Spinodal_Cv_R.csv', usecols = [0,1], names = ['rho','T'] , header = None, delimiter=' ', skipinitialspace = True, skip_blank_lines=True,skiprows = 1) #READ THE FILE AND TRANSFER DATA TO DATAFRAME

dots=data2[data2['T']==tempt]
val ='%.3f'%(float(dots['rho']))
sufix=str('cluster_sys-D'+str(val)+'-T'+str(tempt)+'.data');
filename = str(sufix);
print(filename)
data = pd.read_csv(filename, usecols = [0,1,2,3,4], names = ['T','N','KE','PE','Temp'] , header = None, skiprows = 1, delimiter=' ', skipinitialspace = True, skip_blank_lines=False) #READ THE FILE AND TRANSFER DATA TO A DATA FRAME
a=data[data['T']>Tskip]
x = np.array(a['PE'])
kwargs = dict(histtype='step',density=True,bins=50,alpha=1,lw=2,color='red',label=r'$\rho$='+str(val)) #
z,w,_=plt.hist(x, **kwargs)
print(float(dots['rho']))
cbar.ax.plot([0, 1], [float(dots['rho'])/Maxdata]*2, 'red')

plt.title('#Potential Energy Histogram')
plt.ylim(0,25)
plt.legend(loc="upper center")
plt.xlabel(r'Potential Energy $\left(E*\right)$')
plt.ylabel(r'Frequency')
plt.grid(True)
#plt.tight_layout()
filename3 = "PE_hist_T"+str(tempt)+".png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
