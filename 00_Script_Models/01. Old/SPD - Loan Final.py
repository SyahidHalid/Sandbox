#Fix Stationarity
#https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/

import pandas as pd #data manipulation and analysis
import numpy as np #linear Algebra
import matplotlib.pyplot as plt #data visualisation
import seaborn as sns # data visualisation
import warnings #ignore warnings
from statsmodels.tsa.stattools import adfuller
from sklearn import linear_model

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None) 
pd.set_option("display.max_colwidth", 1000) #huruf dlm column
pd.set_option("display.max_rows", 100)
pd.set_option("display.precision", 2) #2 titik perpuluhan

#Internal = pd.read_excel(r"C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Analytics\Loan Growth Model\2024\Loan Growth - Working.xlsx", sheet_name='Internal')
#External = pd.read_excel(r"C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Analytics\Loan Growth Model\2024\Loan Growth - Working.xlsx", sheet_name='External')

#-------------------------------------------------------------------V4
v4 = pd.read_excel(r"C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Analytics\Loan Growth Model\2024\Loan Growth - Working.xlsx", sheet_name='v4')
#YoY = pd.read_excel(r"C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Analytics\Loan Growth Model\2024\Loan Growth - Working.xlsx", sheet_name='YoY')


v4['YoY Growth'] = v4["Total Outstanding Amount"].pct_change(periods=1) * 100

indepen = v4.iloc[np.where(~v4['YoY Growth'].isna())][['YoY Growth','Application','Disbursed','Front Office','Back Office','Staff Strength','Fed Funds','MY GDP','US GDP','FX','USD Index','Bursa','OPR']]

plt.figure(figsize=(10,6))
sns.heatmap(indepen.corr(),annot=True,square=True)
plt.title('Correlation', size=18);
plt.xticks(size=13)
plt.yticks(size=13)
plt.show()

sns.pairplot(indepen,
                 markers="+",
                diag_kind='kde',
                kind='reg',
                plot_kws={'line_kws':{'color':'#aec6cf'},'scatter_kws':{'alpha':0.7,'color':'red'}},corner=True);

#Differencing
    
indepen['YoY Growth (DF)'] = (indepen['YoY Growth'] - indepen['YoY Growth'].shift(1)).fillna(0)
indepen['YoY Growth (DF2)'] = (indepen['YoY Growth (DF)'] - indepen['YoY Growth (DF)'].shift(1)).fillna(0)

indepen['Application (DF)'] = (indepen['Application'] - indepen['Application'].shift(1)).fillna(0)
indepen['Application (DF2)'] = (indepen['Application (DF)'] - indepen['Application (DF)'].shift(1)).fillna(0)

indepen['Disbursed (DF)'] = (indepen['Disbursed'] - indepen['Disbursed'].shift(1)).fillna(0)
indepen['Disbursed (DF2)'] = (indepen['Disbursed (DF)'] - indepen['Disbursed (DF)'].shift(1)).fillna(0)
indepen['Disbursed (DF3)'] = (indepen['Disbursed (DF2)'] - indepen['Disbursed (DF2)'].shift(1)).fillna(0)
indepen['Disbursed (DF4)'] = (indepen['Disbursed (DF3)'] - indepen['Disbursed (DF3)'].shift(1)).fillna(0)
indepen['Disbursed (DF5)'] = (indepen['Disbursed (DF4)'] - indepen['Disbursed (DF4)'].shift(1)).fillna(0)
indepen['Disbursed (DF6)'] = (indepen['Disbursed (DF5)'] - indepen['Disbursed (DF5)'].shift(1)).fillna(0)

indepen['Staff Strength (DF)'] = (indepen['Staff Strength'] - indepen['Staff Strength'].shift(1)).fillna(0)

indepen['US GDP (DF)'] = (indepen['US GDP'] - indepen['US GDP'].shift(1)).fillna(0)
indepen['US GDP (DF2)'] = (indepen['US GDP (DF)'] - indepen['US GDP (DF)'].shift(1)).fillna(0)

indepen['FX (DF)'] = (indepen['FX'] - indepen['FX'].shift(1)).fillna(0)

indepen['USD Index (DF)'] = (indepen['USD Index'] - indepen['USD Index'].shift(1)).fillna(0)

indepen['OPR (DF)'] = (indepen['OPR'] - indepen['OPR'].shift(1)).fillna(0)
indepen['OPR (DF2)'] = (indepen['OPR (DF)'] - indepen['OPR (DF)'].shift(1)).fillna(0)
indepen['OPR (DF3)'] = (indepen['OPR (DF2)'] - indepen['OPR (DF2)'].shift(1)).fillna(0)

#Check Stationary
ikan = "OPR (DF3)"
X = indepen[ikan].values
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

#Not Stationary YoY Growth
#Statinary is Bursa, MY GDP, Fed Funds



#model

#1st run Buang Disbursed cause above 0.5

Xs = indepen[['Application (DF2)',
              'Fed Funds','MY GDP','US GDP (DF2)','FX (DF)',
              'Bursa','OPR (DF3)','USD Index (DF)','Staff Strength (DF)']]
ys = indepen["YoY Growth"]

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
model.summary()


#Prediction
intercept = -576.5176836896383
x1 = -0.97710259*indepen["Application (DF2)"]
x2 = 8.96921868*indepen["Fed Funds"]
x3 = 0.44174749*indepen["MY GDP"]
x4 = -1.12202745*indepen["US GDP (DF2)"]
x5 = 39.32788799*indepen["FX (DF)"]
x6 = 0.36833983*indepen["Bursa"]
x7 = 56.92625077*indepen["OPR (DF3)"]
x8 = -1.32708837*indepen["USD Index (DF)"]
x9 = -0.41011346*indepen["Staff Strength (DF)"]

indepen["Growth Predicted"] = intercept + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9

#Graph

plt.figure(figsize=(10,4))
plt.plot(indepen["YoY Growth"], color='b')
plt.plot(indepen["Growth Predicted"], color='r')

plt.legend(['YoY Growth','Growth Predicted'], fontsize=8)

plt.show()




#as of Latest Q1 2024
#end of year nak naik 10% so sebulan 5%
ten = 25
application = 53
Fed = 5.25
MY = 3.4
US = 3.1
FX = 4.7
Bursa = 1447.04
OPR = 3
USD = 104.53
Staff = 324

a1 = -0.97710259*application
a2 = 8.96921868*Fed
a3 = 0.44174749*MY
a4 = -1.12202745*US
a5 = 39.32788799*FX
a6 = 0.36833983*Bursa
a7 = 56.92625077*OPR
a8 = -1.32708837*USD
a9 = -0.41011346*Staff

newx1 = (ten - (intercept+a2+a3+a4+a5+a6+a7+a8+a9))/(a1/application)
print('application '+str(newx1))

newx2 = (ten - (intercept+a1+a3+a4+a5+a6+a7+a8+a9))/(a2/Fed)
print('Fed '+str(newx2))

newx3 = (ten - (intercept+a1+a2+a4+a5+a6+a7+a8+a9))/(a3/MY)
print('MY GDP '+str(newx3))

newx4 = (ten - (intercept+a1+a2+a3+a5+a6+a7+a8+a9))/(a4/US)
print('US GDP '+str(newx4))

newx5 = (ten - (intercept+a1+a2+a3+a4+a6+a7+a8+a9))/(a5/FX)
print('FX '+str(newx5))

newx6 = (ten - (intercept+a1+a2+a3+a4+a5+a7+a8+a9))/(a6/Bursa)
print('Bursa '+str(newx6))

newx7 = (ten - (intercept+a1+a2+a3+a4+a5+a6+a8+a9))/(a7/OPR)
print('OPR '+str(newx7))

newx8 = (ten - (intercept+a1+a2+a3+a4+a5+a6+a7+a9))/(a8/USD)
print('USD '+str(newx8))

newx9 = (ten - (intercept+a1+a2+a3+a4+a5+a6+a7+a8))/(a9/Staff)
print('Staff '+str(newx9))
