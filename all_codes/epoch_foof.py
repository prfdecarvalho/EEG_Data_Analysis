import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io
from fooof import FOOOF
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

fname = 'ON_Day1-epo'#'OFF_Day1-epo.fif'

#print(mne.what(fname))
mnedata = mne.read_epochs(fname+'.fif')

df = mne.Epochs.to_data_frame(mnedata)
#df.to_csv('ON_Day1-epo.csv')

nome_col=list(df.columns)
#print(df['epoch'])
fig, ax = plt.subplots(11,1)


# Initialize a FOOOF object
fm = FOOOF()

# Set the frequency range to fit the model
freq_range = [2, 40]


#PSD Log Y HTL
for i in range(1,3):
    unfiltered = df.iloc[:,i+3]
    out = notch(unfiltered, interference_freq, fs)
    f, Pxx_den = signal.welch(out, fs, nperseg=npers)
    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    fm.report(f, Pxx_den, freq_range)
    #plt.semilogy(f, Pxx_den,'.-')
    #plt.xlim(right=hf)

plt.xlabel('frequency [Hz]')
#plt.ylabel(r'PSD [$V^{2}/Hz$]')

filename3 = fname+"_foof.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
##

plt.clf()
