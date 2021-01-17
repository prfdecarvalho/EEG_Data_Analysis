
import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io

fname = 'ON_Day1-epo'#'OFF_Day1-epo.fif'

#print(mne.what(fname)) #find what type of archive is
mnedata = mne.read_epochs(fname+'.fif') #put data on code
df = mne.Epochs.to_data_frame(mnedata) #convert to Dataframe (Pandas)
nome_col=list(df.columns) #list with column names
#df.to_csv('ON_Day1-epo.csv') #Output in CSV

fs = 1024 #freq. threshold
npers = 1024 #resolution
hf = 100 #higher freq. to show

#PSD Log Y HTL
for i in range(1,12):
    f, Pxx_den = signal.welch(df.iloc[:,i+3], fs, nperseg=npers) #
    plt.semilogy(f, Pxx_den,'.-')#plot(df.iloc[:,i+3],lw=0.2)
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')

filename3 = fname+"_PSD_HTL.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
##

plt.clf()

#PSD Log X HTL
for i in range(1,12):
    f, Pxx_den = signal.welch(df.iloc[:,i+3], fs, nperseg=npers)
    plt.semilogx(f, Pxx_den,'.-')#plot(df.iloc[:,i+3],lw=0.2)
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')

filename3 = fname+"_LX_PSD_HTL.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
##

plt.clf()

#PSD Log Y HTR
for i in range(1,10):
    f, Pxx_den = signal.welch(df.iloc[:,i+13], fs, nperseg=npers)
    plt.semilogy(f, Pxx_den,'.-')
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')

filename3 = fname+"_PSD_HTR.png"
plt.savefig(filename3, dpi=500)
##

plt.clf()

#PSD Log X HTR
for i in range(1,10):
    f, Pxx_den = signal.welch(df.iloc[:,i+13], fs, nperseg=npers)
    plt.semilogx(f, Pxx_den,'.-')#plot(df.iloc[:,i+3],lw=0.2)
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')

filename3 = fname+"_LX_PSD_HTR.png"
plt.savefig(filename3, dpi=500)
##

plt.clf()

#PSD Log-Log HTR
for i in range(1,10):
    f, Pxx_den = signal.welch(df.iloc[:,i+13], fs, nperseg=npers)
    plt.loglog(f, Pxx_den,'.-')#plot(df.iloc[:,i+3],lw=0.2)
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')

filename3 = fname+"LL_PSD_HTR.png"
plt.savefig(filename3, dpi=500)
##
plt.clf()

#PSD Log-Log HTL
for i in range(1,12):
    f, Pxx_den = signal.welch(df.iloc[:,i+3], fs, nperseg=npers)
    plt.loglog(f, Pxx_den,'.-')#plot(df.iloc[:,i+3],lw=0.2)
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')

filename3 = fname+"LL_PSD_HTL.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE

plt.clf()

#PSD Log-Log HTR
for i in range(1,10):
    f, Pxx_den = signal.welch(df.iloc[:,i+13], fs, nperseg=npers)
    plt.plot(f, Pxx_den,'.-')#plot(df.iloc[:,i+3],lw=0.2)
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')

filename3 = fname+"Lin_PSD_HTR.png"
plt.savefig(filename3, dpi=500)
##
plt.clf()

#PSD Linear HTL

for i in range(1,12):
    f, Pxx_den = signal.welch(df.iloc[:,i+3], fs, nperseg=npers)
    plt.plot(f, Pxx_den,'.-')#plot(df.iloc[:,i+3],lw=0.2)
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')

filename3 = fname+"Lin_PSD_HTL.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
