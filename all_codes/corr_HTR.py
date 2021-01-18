import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io
from fooof import FOOOF
import seaborn as sn
import pandas as pd


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
fname = 'OFF_Day2-epo'#file prefix
mnedata = mne.read_epochs(fname+'.fif')
df = mne.Epochs.to_data_frame(mnedata)
nome_col=list(df.columns)
heads=nome_col[14:23]
print(heads)
nchans=9
zipped=zip(*[df.iloc[:,col] for col in range(3,nchans+3)])

chans_data=pd.DataFrame(zipped, columns=heads)
print(chans_data)
#YourMatrix = df.copy()
dcorr = chans_data.corr()
MatrixCorr_aux=pd.DataFrame(dcorr)
mask = np.zeros_like(MatrixCorr_aux)
mask[np.triu_indices_from(mask)] = True

figuresns=sn.heatmap(dcorr,mask=mask,square=True,annot=True,cmap="RdBu_r",cbar=True,cbar_kws={'shrink':0.65})#, ax=ax1),,vmin=0,vmax=.5

filename3 = fname+"_corr_HTR.png"
plt.savefig(filename3, dpi=500)
#MatrixCorr_aux=pd.DataFrame(PairCorr_overtime[tt,:,:])
#df=df.drop(columns=[0,1,2])
#print(df)
#plt.matshow(df.corr())
#plt.show()
