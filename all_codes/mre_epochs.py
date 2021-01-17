import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io
import mrestimator as mre
from matplotlib.font_manager import FontProperties


fname = 'OFF_Day1-epo'#'OFF_Day1-epo.fif'

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
#unfiltered = df.iloc[:,6]
nchans=11
nepochs=99
mrs=np.zeros(shape=(nepochs,nchans))
epochs=np.zeros(shape=(nepochs))
for j in range(1,nchans+1):
    for i in range(1,nepochs+1): #take coeff for each epoch
        data = df.copy()
        filter = data['epoch'] == i
        data.where(filter, inplace = True)#[:,i+3]
        df2 = data.iloc[:,j+2]
        df2 = df2[df2.notna()]
        unfiltered = df2
        x = notch(unfiltered, interference_freq, fs)
        bp = prepared = mre.input_handler(x)
        rk = mre.coefficients(x,method='ts',dtunit='step')

        # compare the builtin fitfunctions
        m1 = mre.fit(rk, fitfunc=mre.f_exponential)
        mrs[i-1,j-1]=m1.mre
        epochs[i-1]=i

    plt.plot(epochs,mrs,label=nome_col[j+2])
fontP = FontProperties()
fontP.set_size('xx-small')
plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
plt.title('mr estimator each epoch')
plt.xlabel('epoch')
plt.ylabel('m')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
filename3 = fname+"_mrs.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
##

zipped=zip(epochs,*[mrs[:,col] for col in range(0,nchans)])
df = pd.DataFrame(zipped)
print(df)
#df=df[(df != 0).all(0)]
heads=nome_col[2:14]
print(heads)
#heads.insert(0, "epochs")
#print(heads)
csvname=str(fname)+'_mr_data.csv'
df.to_csv(csvname, mode='w+',index=None, header=heads,sep =' ')

plt.clf()
