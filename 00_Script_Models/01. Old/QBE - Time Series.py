import numpy as np
import pandas as pd


pd.set_option("display.max_columns", None) 
pd.set_option("display.max_colwidth", 1000) #huruf dlm column
pd.set_option("display.max_rows", 100)
pd.set_option("display.precision", 2) #2 titik perpuluhan

#AA = pd.read_excel(r'C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Syahid Info\Resume\QBE\AA- working.xlsx', sheet_name = 'Sheet1')


CALL = pd.read_csv(r"C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Syahid Info\Resume\QBE\case_data_calls.csv", sep = ";", header = 0)
RESER = pd.read_csv(r"C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Syahid Info\Resume\QBE\case_data_reservations.csv", sep = ";", header = 0)

#---------------------------------------------------------------------------------

AA = CALL.merge(RESER,on='date', how='left')
AA.shape
#.rename(columns={'Account_No':'M_Account_No'})
#AA.to_excel(r'C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Syahid Info\Resume\QBE\AA- working.xlsx', index=False)


Call = AA[['date','calls','weekday','Year','Year Month','Staff Needed Per Day']]
Reser = AA.iloc[np.where(~AA.total_reservations.isna())][['date','reservations_2months_advance','total_reservations','summer_break',
            'christmas_break','special_day','Year','Year Month']]

Call.info()
Call['date'].value_counts()
Call.describe()
CALL.describe(include='O') #for object
CALL.isnull().sum()
CALL[CALL.duplicated]

RESER.info()
RESER['date'].value_counts()
RESER.describe()
RESER.describe(include='O') #for object
RESER.isnull().sum()

corr = AA.iloc[np.where(~AA.total_reservations.isna())].drop(['date',
            'weekday','summer_break',
            'christmas_break','special_day'], axis=1)

#[['date','calls','total_reservations','summer_break',
#            'christmas_break','special_day','Year','Year Month']]

#correlation
import matplotlib.pyplot as plt #data visualisation
import seaborn as sns # data visualisation

sns.pairplot(corr,
                 markers="+",
                diag_kind='kde',
                kind='reg',
                plot_kws={'line_kws':{'color':'#aec6cf'},'scatter_kws':{'alpha':0.7,'color':'red'}},corner=True);


plt.figure(figsize=(10,6))
sns.heatmap(corr.corr(),annot=True,square=True)
plt.title('Correlation', size=18);
plt.xticks(size=13)
plt.yticks(size=13)
plt.show()


#Ho(Accepted): Sample is from the normal distributions.(Po>0.05)
#Ha(Rejected): Sample is not from the normal distributions.

from scipy.stats import shapiro
shapiro(corr['calls'])
shapiro(corr['reservations_2months_advance'])
shapiro(corr['total_reservations'])
#shapiro(corr['Staff Needed Per Day'])
#Since in the above example, the p-value is 0.73 which is more than the threshold(0.05) which is the alpha(0.05) then we fail to reject the null hypothesis i.e. we do not have sufficient evidence to say that sample does not come from a normal distribution.



#Check Stationary
from statsmodels.tsa.stattools import adfuller
ikan = "total_reservations_(DF)"
X = corr[ikan].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

if result[0] < result[4]["5%"]:
    print (ikan+" Reject Ho - Time Series is Stationary")
else:
    print (ikan+" Failed to Reject Ho - Time Series is Non-Stationary")

#Differencing
    
corr['calls_(DF)'] = (corr['calls'] - corr['calls'].shift(1)).fillna(0)
corr['reservations_2months_advance_(DF)'] = (corr['reservations_2months_advance'] - corr['reservations_2months_advance'].shift(1)).fillna(0)
corr['total_reservations_(DF)'] = (corr['total_reservations'] - corr['total_reservations'].shift(1)).fillna(0)

#Linear
#from sklearn.linear_model import LinearRegression

#xa = np.array(corr['calls']).reshape((-1, 1))
#ya = np.array(corr["Staff Needed Per Day"]).reshape((-1, 1))
#odel = LinearRegression()
#odel.fit(xa, ya)
#print(odel.intercept_, odel.coef_, odel.score(xa, ya))

#multiple
from sklearn import linear_model
Xs = corr[['reservations_2months_advance_(DF)','total_reservations_(DF)']]
ys = corr["calls_(DF)"]

regr = linear_model.LinearRegression()
regr.fit(Xs, ys)

print(regr.intercept_)
print(regr.coef_)
print(regr.score(Xs, ys))

import statsmodels.api as sm

#add constant to predictor variables
xs = sm.add_constant(Xs)

#fit linear regression model
model = sm.OLS(ys, xs).fit()

#view model summary
print(model.summary())



#Prediction

#example nk 15% growth 5400706.48015064
#df_i_corr.tail(1)

#y = 0.4959 + 0.0331b1 + 0.1589b2


b1 = 4559
b2 = 27690

b1 = 2340
b2 = 12658

b1 = 965
b2 = 6507

b1 = 2915
b2 = 18435

b1 = 4035
b2 = 25819

b1 = 5662
b2 = 34497

b1 = 17681
b2 = 79727

b1 = 8059
b2 = 45297

b1 = 4820
b2 = 24777

b1 = 4308
b2 = 17111

callV = (0.4959 + (0.0331*b1) + (0.1589*b2))#/80
HeadV = (callV)/80
print(callV)
print(HeadV)


#satu org = 80 calls

AdvRerCoef = 0.03314827
TotRerCoef = 0.15892685


intercept = +0.49594209329999295
x1 = -18.64739986*AdvRerCoef
x2 = -12.1191211*TotRerCoef

fifteen = 10

newNT = (fifteen - (intercept+x2+x3+x4+x5+x6+x7))/(x1/NT)
print(newNT)

newT = (fifteen - (intercept+x1+x3+x4+x5+x6+x7))/(x2/T)
print(newT)

newP = (fifteen - (intercept+x1+x2+x4+x5+x6+x7))/(x3/Pending)
print(newP)

newD = (fifteen - (intercept+x1+x2+x3+x5+x6+x7))/(x4/Disbrused)
print(newD)

newC= (fifteen - (intercept+x1+x2+x3+x4+x6+x7))/(x5/Cancelled)
print(newC)

newW= (fifteen - (intercept+x1+x2+x3+x4+x5+x7))/(x6/WriteOff)
print(newW)

newS= (fifteen - (intercept+x1+x2+x3+x4+x5+x6))/(x7/Settled)
print(newS)