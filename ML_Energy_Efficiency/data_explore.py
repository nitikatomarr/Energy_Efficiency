import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from ucimlrepo import fetch_ucirepo

energy_efficiency = fetch_ucirepo(id=242)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets

df = pd.concat([X, y], axis=1)
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.duplicated().sum())

print(df.corr())            # corr() returns values between -1 and 1
# 1 = perfect positive correlation, -1 = perfect negative, 0 = no correlation

sns.heatmap(df.corr(), 
            cmap = 'coolwarm',      # blue = negative correlation, red = positive
            annot=True,
            fmt = ".2f",
            annot_kws = {"size":7} )       
plt.show()

plt.subplot(1, 2, 1)
sns.histplot(data=df['Y1'])   # plot distribution of Heating Load
plt.xlabel('Heating Load')
plt.title('Assessing the Heating load requirements',
          fontsize = 6)

plt.subplot(1, 2, 2)
sns.histplot(data=df['Y2'])   # plot distribution of Cooling Load
plt.title('Assessing the Cooling load requirements',
          fontsize = 6)
plt.xlabel('Cooling Load')
plt.show()

print(df['X5'].unique())
# X5 (Overall Height) has only 2 unique values: 3.5 and 7
# This explains the bimodal distribution in Y1 and Y2
# Tall vs short buildings have very different energy requirements

print(df['X4'].unique())
# X4 (Roof Area) was suspicious because its 75th percentile equalled its max
# Investigating shows it has 4 unique values
