
import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fname = 'ON_Day1-epo.fif'#'OFF_Day1-epo.fif'

print(mne.what(fname))
mnedata = mne.read_epochs(fname)
print(mnedata)
#events = mne.find_events(fname)

#show in matplotlb
df = mne.Epochs.to_data_frame(mnedata)
df.to_csv('ON_Day1-epo.csv')

plt.plot(df.HTL01)
plt.show()

#events = mne.find_events(fname)
print(df)
mne.Epochs.plot(mnedata)

#epoch= mne.Epochs(mnedata,events)

#raw = mne.io.read_raw_fif(fname, allow_maxshield=True) #op.join(data_path, fname))
