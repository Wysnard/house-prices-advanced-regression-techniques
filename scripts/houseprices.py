import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import norm, skew

train_df = pd.read_csv(r'../input/train.csv')
train_df['Source'] = 0
test_df = pd.read_csv(r'../input/test.csv')
test_df['Source'] = 1

corr = train_df.corr()
corr.to_csv('corr.csv')

train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

sns.distplot(train_df['SalePrice'], fit=norm)
(mu, sigma) = norm.fit(train_df['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()

train_df = train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)].index)
y_train = train_df['SalePrice']
train_df = train_df.drop(['SalePrice'], axis=1)

df = pd.concat([train_df, test_df])
LotFgrp = df.groupby('Neighborhood')['LotFrontage'].median()
v = {'MSZoning' : df['MSZoning'].mode()[0], 'BsmtQual' : 'NA',
	'BsmtCond' : 'NA', 'BsmtExposure' : 'NA', 'BsmtFinType1' : 'NA',
	'BsmtFinSF1' : 0, 'BsmtFinType2' : 'NA', 'BsmtFinSF2' : 0,
	'BsmtUnfSF' : 0, 'TotalBsmtSF' : 0, 'BsmtFullBath' : 0,
	'BsmtHalfBath' : 0, 'LotFrontage' : df['Neighborhood'].map(LotFgrp),
	'GarageArea' : 0, 'GarageCars' : 0, 'GarageYrBlt' : 0, 'MasVnrArea' : 0}
df.fillna(value=v, inplace=True)
df['MSSubClass'] = df['MSSubClass'].apply(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(df[c].values))
    df[c] = lbl.transform(list(df[c].values))

des = df.describe()
des.to_csv('describe.csv')
df.to_csv('house.csv')
