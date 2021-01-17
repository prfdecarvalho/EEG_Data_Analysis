
import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy import signal

fname = 'OFF_Day1-epo.fif'#'OFF_Day1-epo.fif'

#print(mne.what(fname))
mnedata = mne.read_epochs(fname)

df = mne.Epochs.to_data_frame(mnedata)
#df.to_csv('ON_Day1-epo.csv')

nome_col=list(df.columns)
#print(nome_col)
fig, ax = plt.subplots(11,1)

for i in range(1,12):
    ax[i-1].plot(df.iloc[:,i+3],lw=0.2)
    ax[i-1].set_ylabel(nome_col[i+3],size=10)
    ax[i-1].set_yticklabels([])
    if i<11:
        ax[i-1].set_xticklabels([])
    ax[i-1].set_xlim(6000,16000)

ax[10].set_xlabel(r'Time (ms)')
#plt.title('Timeseries')
filename3 = fname+"_TS_HTL.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE

plt.clf()

fig, ax = plt.subplots(9,1)

for i in range(1,10):
    ax[i-1].plot(df.iloc[:,i+13],lw=0.2)
    #print(nome_col[i+13])
    ax[i-1].set_ylabel(nome_col[i+13],size=10)
    ax[i-1].set_yticklabels([])
    if i<9:
        ax[i-1].set_xticklabels([])
    ax[i-1].set_xlim(6000,16000)

ax[8].set_xlabel(r'Time (ms)')
#plt.title('Timeseries')
filename3 = fname+"_TS_HTR.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
