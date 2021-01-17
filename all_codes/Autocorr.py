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
interference_freq = 50 #NOISE
#############

#PSD Log Y HTL
for i in range(2,4):
    unfiltered = df.iloc[:,i+5]
    x = notch(unfiltered, interference_freq, fs)
    #result = np.correlate(out, out, mode='full')
    #plt.plot(np.arange(len(result)), result)
    nx = len(x)
    lags = np.arange(-nx + 1, nx)/fs

    # Remove sample mean.
    xdm = x - np.mean(x)
    #ydm = y - np.mean(y)

    ccov = np.correlate(xdm, xdm, mode='full') #correlation
    ccor = ccov / (nx * xdm.std() * xdm.std()) #Normalization

    plt.plot(lags,ccor,label=nome_col[i+5])
    plt.xlim(-2.5,2.5)
    plt.legend()
    #plt.show()
plt.grid(True)

#plt.show()

#plt.xlabel('frequency [Hz]')
#plt.ylabel(r'PSD [$V^{2}/Hz$]')

filename3 = fname+"acorr.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
##

plt.clf()
