import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io
import mrestimator as mre
from matplotlib.font_manager import FontProperties


fname = 'ON_Day2-epo'#'OFF_Day1-epo.fif'

#print(mne.what(fname)) #find what type of archive is
mnedata = mne.read_epochs(fname+'.fif') #put data on code
df = mne.Epochs.to_data_frame(mnedata) #convert to Dataframe (Pandas)
nome_col=list(df.columns) #list with column names

csvname=str(fname)+'_mr_data_HTR.csv'
data = pd.read_csv(csvname, delimiter=' ') #READ THE FILE AND TRANSFER DATA TO A DATA FRAME
print(data.columns[1])
for j in range(1,len(data.columns)):
    print(j)
    plt.plot(data.iloc[:,0],data.iloc[:,j],label=data.columns[j])

fontP = FontProperties()
fontP.set_size('xx-small')
plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
plt.title('mr estimator each epoch; file: '+fname)
plt.xlabel('epoch')
plt.ylabel('m')
#plt.ylim(0.95,1.0)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
filename3 = fname+"_mrs_HTR.png"
plt.savefig(filename3, dpi=500) #EXPORT PLOT AS PNG FILE
