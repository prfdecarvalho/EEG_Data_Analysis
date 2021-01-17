
import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fname = 'ON_Day1-epo.fif'#'OFF_Day1-epo.fif'

print(mne.what(fname))
mnedata = mne.read_epochs(fname)
crop_mne = mnedata.crop(0.5,0.51)
print(crop_mne)
#channs = mne.Epochs.pick_channels('HTR07',mnedata)
#print(channs)
#events = mne.find_events(mnedata,stim_channel='HTR07')

####show in matplotlib####
#df = mne.Epochs.to_data_frame(mnedata)
#df.to_csv('ON_Day1-epo.csv')
#plt.plot(df.HTL01)
#plt.show()
##########################

#print(range(1,99))
#print(df)
#raw = mne.io.read_raw_fif(mnedata)
#epochs = mne.Epochs(mnedata,np.arange(1,20))
mne.Epochs.plot(crop_mne,np.arange(1,20))
plt.show()

#epoch= mne.Epochs(mnedata,events)

#raw = mne.io.read_raw_fif(fname, allow_maxshield=True) #op.join(data_path, fname))
