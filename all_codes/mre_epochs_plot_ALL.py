import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io
import mrestimator as mre
from matplotlib.font_manager import FontProperties


fname = 'OFF_Day1-epo'#'OFF_Day1-epo.fif'
csvname=str(fname)+'_mr_data_HTR.csv'
data = pd.read_csv(csvname, delimiter=' ') #READ THE FILE AND TRANSFER DATA TO A DATA FRAME
m_off_hr = np.zeros(9)

for j in range(1,len(data.columns)):
    print(j)
    #plt.plot(data.iloc[:,0],data.iloc[:,j],label=data.columns[j])
    m_off_hr[j-1] = np.var(data.iloc[:,j])

csvname=str(fname)+'_mr_data_HTL.csv'
data = pd.read_csv(csvname, delimiter=' ') #READ THE FILE AND TRANSFER DATA TO A DATA FRAME
m_off_hl = np.zeros(11)

for j in range(1,len(data.columns)):
    print(j)
    #plt.plot(data.iloc[:,0],data.iloc[:,j],label=data.columns[j])
    m_off_hl[j-1] = np.var(data.iloc[:,j])


fname = 'ON_Day1-epo'#'OFF_Day1-epo.fif'
csvname=str(fname)+'_mr_data_HTR.csv'
data = pd.read_csv(csvname, delimiter=' ') #READ THE FILE AND TRANSFER DATA TO A DATA FRAME
m_on_hr = np.zeros(9)

for j in range(1,len(data.columns)):
    print(j)
    #plt.plot(data.iloc[:,0],data.iloc[:,j],label=data.columns[j])
    m_on_hr[j-1] = np.var(data.iloc[:,j])

csvname=str(fname)+'_mr_data_HTL.csv'
data = pd.read_csv(csvname, delimiter=' ') #READ THE FILE AND TRANSFER DATA TO A DATA FRAME
m_on_hl = np.zeros(11)

for j in range(1,len(data.columns)):
    print(j)
    #plt.plot(data.iloc[:,0],data.iloc[:,j],label=data.columns[j])
    m_on_hl[j-1] = np.var(data.iloc[:,j])

plt.scatter(np.zeros(len(m_off_hr)), m_off_hr,label='CONTROL HRT',marker='o',color='orange')
plt.scatter(np.ones(len(m_on_hr)), m_on_hr,label='DRUG HRT',marker='o',color='blue')
plt.scatter(np.zeros(len(m_off_hl)), m_off_hl,label='CONTROL HRL',marker='x',color='orange')
plt.scatter(np.ones(len(m_on_hl)), m_on_hl,label='DRUG HRL',marker='x',color='blue')
for i in range(9):
    x=[0,1]
    y=[m_off_hr[i],m_on_hr[i]]
    plt.plot(x, y,lw=0.5,color='black')

for i in range(11):
    x=[0,1]
    y=[m_off_hl[i],m_on_hl[i]]
    plt.plot(x, y,lw=0.5,color='black')

fontP = FontProperties()
fontP.set_size('xx-small')
plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
plt.title('mr estimator each channel')
plt.xlabel('')
plt.ylabel('variance of m')
plt.xlim(-1,2)
plt.ylim(0,0.00046)
plt.xticks([])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
filename3 = fname+"_mr_CHAN_all_var.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
