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
ax = plt.subplot(111)
#PARAMETERS#
fs = 1024 #freq. threshold
npers = 1024 #resolution
hf = 100 #higher freq. to show
interference_freq = 50 #NOISE
#############

####GET ON###
fname = 'ON_Day2-epo'#file prefix
mnedata = mne.read_epochs(fname+'.fif')
df = mne.Epochs.to_data_frame(mnedata)
nome_col=list(df.columns)

# Initialize a FOOOF object
fm = FOOOF()
# Set the frequency range to fit the model
freq_range = [2, 40]

nr2 = np.zeros(20) #R² exponent

############################
#PSD Log Y HTL
for i in range(20):
    unfiltered = df.iloc[:,i+3]
    out = notch(unfiltered, interference_freq, fs)
    f, Pxx_den = signal.welch(out, fs, nperseg=npers)
    fm.fit(f, Pxx_den, freq_range)
    nr2[i] = -1 * fm.aperiodic_params_[1]

plt.scatter(np.ones(len(nr2)), nr2,label='DRUG')

####GET OFF###
fname = 'OFF_Day2-epo'#file prefix
mnedata = mne.read_epochs(fname+'.fif')
df = mne.Epochs.to_data_frame(mnedata)
nome_col=list(df.columns)

# Initialize a FOOOF object
fm = FOOOF()
# Set the frequency range to fit the model
freq_range = [2, 40]
nr22 = np.zeros(20) #R² exponent

############################
for i in range(20):
    unfiltered = df.iloc[:,i+3]
    print(nome_col[i+3])
    out = notch(unfiltered, interference_freq, fs)
    f, Pxx_den = signal.welch(out, fs, nperseg=npers)
    fm.fit(f, Pxx_den, freq_range)
    nr22[i] = -1 *fm.aperiodic_params_[1]

plt.scatter(np.zeros(len(nr22)), nr22,label='CONTROL')

for i in range(20):
    x=[1,0]
    y=[nr2[i],nr22[i]]
    plt.plot(x, y,lw=0.5,color='black')
    ax.annotate(nome_col[i+3], (x[0], y[0]))
    ax.annotate(nome_col[i+3], (x[1], y[1]))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xlim(-1,2)
plt.xticks([])
plt.legend()
plt.title('1/f Exponent')
plt.ylabel(r'$\beta \to (1/f^{\beta})$')

filename3 = fname+"CHAN_foof.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
