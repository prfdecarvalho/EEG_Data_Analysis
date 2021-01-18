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

df2 = df.iloc[:,8]
unfiltered = df2
x = notch(unfiltered, interference_freq, fs)

dfa = pd.DataFrame(x)
csvname=fname+'_'+str(nome_col[8])+'_dfa.csv'
dfa.to_csv(csvname, mode='w+',index=None, header=nome_col[8],sep =' ')
csvname=fname+'_'+str(nome_col[8])+'_dfa.dat'
dfa.to_csv(csvname, sep = "|", mode='w+',index=None, header=nome_col[8])
np.savetxt(str(nome_col[8])+'_dfa.dat', dfa.values)#, fmt='%d')

print(dfa)
