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
fname = 'ON_Day1-epo'#file prefix
mnedata = mne.read_epochs(fname+'.fif')
df = mne.Epochs.to_data_frame(mnedata)
nome_col=list(df.columns)

# Initialize a FOOOF object
fm = FOOOF()
# Set the frequency range to fit the model
freq_range = [2, 40]

npeaks = np.zeros(99) #number of peaks
nr2 = np.zeros(99) #R² exponent
epochs = np.zeros(99)

unfiltered = df.iloc[:,8]
out = notch(unfiltered, interference_freq, fs)
f, Pxx_den = signal.welch(out, fs, nperseg=npers)
fm.report(f, Pxx_den, freq_range,plt_log=False)
r2_on_all = -1*fm.aperiodic_params_[1]

plt.title('(DRUG) Fit report for channel '+str(nome_col[8]))
filename3 = fname+"_report.png"
plt.savefig(filename3, dpi=500)

plt.clf()
############################
#PSD Log Y HTL
for i in range(1,100):
    data = df.copy()
    filter = data['epoch'] == i
    data.where(filter, inplace = True)#[:,i+3]
    df2 = data.iloc[:,8]
    df2 = df2[df2.notna()]
    unfiltered = df2
    out = notch(unfiltered, interference_freq, fs)
    f, Pxx_den = signal.welch(out, fs, nperseg=npers)

    fm.fit(f, Pxx_den, freq_range)
    npeaks[i-1] = fm.n_peaks_
    nr2[i-1] = -1*fm.aperiodic_params_[1]
    epochs[i-1] = i


plt.xlabel('frequency [Hz]')
#plt.ylabel(r'PSD [$V^{2}/Hz$]')

filename3 = fname+"_foof.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
##

plt.clf()
####GET OFF###
fname = 'OFF_Day1-epo'#file prefix
mnedata = mne.read_epochs(fname+'.fif')
df = mne.Epochs.to_data_frame(mnedata)
nome_col=list(df.columns)

# Initialize a FOOOF object
fm = FOOOF()
# Set the frequency range to fit the model
freq_range = [2, 40]

npeaks2 = np.zeros(99) #number of peaks
nr22 = np.zeros(99) #R² exponent
epochs2 = np.zeros(99)

unfiltered = df.iloc[:,8]
out = notch(unfiltered, interference_freq, fs)
f, Pxx_den = signal.welch(out, fs, nperseg=npers)
fm.report(f, Pxx_den, freq_range,plt_log=False)
r2_off_all = -1 * fm.aperiodic_params_[1]
plt.xlabel('frequency [Hz]')
plt.title('(CONTROL) Fit report for channel '+str(nome_col[8]))

filename3 = fname+"_foof_report.png"
plt.savefig(filename3, dpi=500)

plt.clf()
############################


#PSD Log Y HTL
for i in range(1,100):
    data = df.copy()
    filter = data['epoch'] == i
    data.where(filter, inplace = True)#[:,i+3]
    df2 = data.iloc[:,8]
    df2 = df2[df2.notna()]
    unfiltered = df2
    out = notch(unfiltered, interference_freq, fs)
    f, Pxx_den = signal.welch(out, fs, nperseg=npers)

    #fm.report(f, Pxx_den, freq_range)
    fm.fit(f, Pxx_den, freq_range)
    npeaks2[i-1] = fm.n_peaks_
    nr22[i-1] = -1 * fm.aperiodic_params_[1]
    epochs2[i-1] = i

filename3 = fname+"_foof.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
##

plt.clf()

plt.scatter(np.ones(len(nr2)), nr2,label='DRUG')
plt.scatter(np.zeros(len(nr22)), nr22,label='CONTROL')
plt.xlim(-1,2)
plt.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.title('1/f Exponent for each epoch, channel '+str(nome_col[8]))
plt.ylabel(r'$\beta \to (1/f^{\beta})$')


filename3 = fname+"R2_foof.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE

plt.clf()

plt.plot(epochs, nr2,label='DRUG')
plt.plot(epochs, nr22,label='CONTROL')
plt.axhline(y=r2_on_all, label='DRUG All',ls='--')
plt.axhline(y=np.mean(nr2), label='Mean DRUG')
plt.axhline(y=r2_off_all, label='CONTROL All',color='orange',ls='--')
plt.axhline(y=np.mean(nr22), label='Mean CONTROL',color='orange')

plt.title('Expoents for each Epoch, channel '+nome_col[8])
plt.legend()
plt.xlabel('epoch')
plt.ylabel(r'$\beta$ exponent')

filename3 = fname+"R2Ep_foof.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
