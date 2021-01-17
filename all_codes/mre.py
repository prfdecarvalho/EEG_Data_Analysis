import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io
import mrestimator as mre


fname = 'ON_Day2-epo'#'OFF_Day1-epo.fif'

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
#for i in range(2,4):
unfiltered = df.iloc[:,6]
x = notch(unfiltered, interference_freq, fs)
bp = prepared = mre.input_handler(x)
#print(mre.simulate_branching(m=0.99, a=10, numtrials=15))
rk = mre.coefficients(x,method='ts',dtunit='step',steps=(1,10000))

# compare the builtin fitfunctions
m1 = mre.fit(rk, fitfunc=mre.f_exponential)
#m2 = mre.fit(rk, fitfunc=mre.f_exponential_offset)
#m3 = mre.fit(rk, fitfunc=mre.f_complex)

# plot manually without using OutputHandler
plt.plot(rk.steps, rk.coefficients, label='data')
plt.plot(rk.steps, mre.f_exponential(rk.steps, *m1.popt),
    label='exponential m={:.5f}'.format(m1.mre))
#plt.plot(rk.steps, mre.f_exponential_offset(rk.steps, *m2.popt),
    #label='exp + offset m={:.5f}'.format(m2.mre))
#plt.plot(rk.steps, mre.f_complex(rk.steps, *m3.popt),
    #label='complex m={:.5f}'.format(m3.mre))

plt.legend()
plt.show()


filename3 = fname+"acorr.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
##

plt.clf()
