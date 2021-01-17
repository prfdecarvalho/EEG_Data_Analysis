from scipy.stats import t
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

csvname='results.dat'
data = pd.read_csv(csvname, delimiter=' ')
x=data[data.iloc[:,0]<2]
print(x)
#Perform the linear regression:print(x)
res = stats.linregress(x.iloc[:,0], x.iloc[:,1])

#Plot the data along with the fitted line:
plt.plot(x.iloc[:,0], x.iloc[:,1], 'o', label='original data')
plt.plot(x.iloc[:,0], res.intercept + res.slope*x.iloc[:,0], 'r', label='fitted line; a='+str(res.slope)+'; b = '+str(res.intercept))
print(res.slope)
print(res.intercept)

plt.title('DFA single channel')
plt.xlabel('n ; box-size')
plt.ylabel('F(n) ; self-affinity')
#plt.ylim(top=8)
plt.legend()

filename3 = "OFF_Day2-epo_HTL06_DFA_Z.png"
plt.savefig(filename3, dpi=500)

plt.show()
