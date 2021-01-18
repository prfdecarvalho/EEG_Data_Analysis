import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io

fname = 'OFF_Day2-epo'#'OFF_Day1-epo.fif'

#print(mne.what(fname)) #find what type of archive is
mnedata = mne.read_epochs(fname+'.fif') #put data on code
df = mne.Epochs.to_data_frame(mnedata) #convert to Dataframe (Pandas)
nome_col=list(df.columns) #list with column names
#df.to_csv('ON_Day1-epo.csv') #Output in CSV

###Filter
def notch(unfiltered, interference_frequency, sampling_frequency, rf=35.):
   # w0 is the interference frequency expressed in cycles/half-cycle. Half-cycle corresponds to the Nyquist frequency
   # w0=1 for the Nyquist frequency (sampling/2)
   w0 = interference_frequency / (sampling_frequency / 2.)
   bw = w0 / rf  # reducing factor=35
   lower = w0 - bw / 2.
   if lower < 0: lower = 0.0000000001
   upper = w0 + bw / 2.
   if upper > 1: upper = 0.9999999999
   b, a = signal.butter(2, [lower, upper], btype='bandstop')
   filtered = signal.lfilter(b, a, unfiltered)

   return filtered
#######

#PARAMETERS#
fs = 1024 #freq. threshold
npers = 1024 #resolution
hf = 100 #higher freq. to show
interference_freq = 300 #NOISE
#############

#PSD Log Y HTL
for i in range(1,12):
    unfiltered = df.iloc[:,i+2]
    out = notch(unfiltered, interference_freq, fs)
    f, Pxx_den = signal.welch(out, fs, nperseg=npers)
    plt.semilogy(f, Pxx_den,label=nome_col[i+2])
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')
plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
plt.title('PSD Log Y HTL; File: '+fname)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
filename3 = fname+"_PSD_HTL_NF.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
##

plt.clf()

#PSD Log X HTL
for i in range(1,12):
    unfiltered = df.iloc[:,i+2]
    out = notch(unfiltered, interference_freq, fs)
    f, Pxx_den = signal.welch(out, fs, nperseg=npers)
    plt.semilogx(f, Pxx_den,label=nome_col[i+2])
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')
plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
plt.title('PSD Log X HTL; File: '+fname)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

filename3 = fname+"_LX_PSD_HTL_NF.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
##

plt.clf()

#PSD Log Y HTR
for i in range(1,10):
    unfiltered = df.iloc[:,i+13]
    out = notch(unfiltered, interference_freq, fs)
    f, Pxx_den = signal.welch(out, fs, nperseg=npers)
    plt.semilogy(f, Pxx_den,label=nome_col[i+13])
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')
plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
plt.title('PSD Log Y HTR; File: '+fname)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

filename3 = fname+"_PSD_HTR_NF.png"
plt.savefig(filename3, dpi=500)
##

plt.clf()

#PSD Log X HTR
for i in range(1,10):
    unfiltered = df.iloc[:,i+13]
    out = notch(unfiltered, interference_freq, fs)
    f, Pxx_den = signal.welch(out, fs, nperseg=npers)
    plt.semilogx(f, Pxx_den,label=nome_col[i+13])
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')
plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
plt.title('PSD Log X HTR; File: '+fname)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

filename3 = fname+"_LX_PSD_HTR_NF.png"
plt.savefig(filename3, dpi=500)
##

plt.clf()

#PSD Log-Log HTR
for i in range(1,10):
    unfiltered = df.iloc[:,i+13]
    out = notch(unfiltered, interference_freq, fs)
    f, Pxx_den = signal.welch(out, fs, nperseg=npers)
    plt.loglog(f, Pxx_den,label=nome_col[i+13])
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')
plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
plt.title('PSD Log-Log HTR; File: '+fname)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

filename3 = fname+"_LL_PSD_HTR_NF.png"
plt.savefig(filename3, dpi=500)
##
plt.clf()

#PSD Log-Log HTL
for i in range(1,12):
    unfiltered = df.iloc[:,i+2]
    out = notch(unfiltered, interference_freq, fs)
    f, Pxx_den = signal.welch(out, fs, nperseg=npers)
    plt.loglog(f, Pxx_den,label=nome_col[i+2])#plot(df.iloc[:,i+3],lw=0.2)
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')
plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
plt.title('PSD Log-Log HTL; File: '+fname)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

filename3 = fname+"_LL_PSD_HTL_NF.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE

plt.clf()

#PSD Linear HTR
for i in range(1,10):
    unfiltered = df.iloc[:,i+13]
    out = notch(unfiltered, interference_freq, fs)
    f, Pxx_den = signal.welch(out, fs, nperseg=npers)
    plt.plot(f, Pxx_den,label=nome_col[i+13])#plot(df.iloc[:,i+3],lw=0.2)
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')
plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
plt.title('PSD Log-Log HTL; File: '+fname)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

filename3 = fname+"_Lin_PSD_HTR_NF.png"
plt.savefig(filename3, dpi=500)
##
plt.clf()

#PSD Linear HTL

for i in range(1,12):
    unfiltered = df.iloc[:,i+2]
    out = notch(unfiltered, interference_freq, fs)
    f, Pxx_den = signal.welch(out, fs, nperseg=npers)
    plt.plot(f, Pxx_den,label=nome_col[i+2])#plot(df.iloc[:,i+3],lw=0.2)
    plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
plt.ylabel(r'PSD [$V^{2}/Hz$]')
plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
plt.title('PSD Log-Log HTL; File: '+fname)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

filename3 = fname+"_Lin_PSD_HTL_NF.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
