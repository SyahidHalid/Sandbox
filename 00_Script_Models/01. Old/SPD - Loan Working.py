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

v4['YoY Growth'] = v4["Total Outstanding Amount"].pct_change(periods=1) * 100
v4['YoY App'] = v4["Application"].pct_change(periods=1) * 100
v4['YoY Dis'] = v4["Disbursed"].pct_change(periods=1) * 100
v4['YoY Staff'] = v4["Staff Strength"].pct_change(periods=1) * 100

indepen = v4[['Total Outstanding Amount','Application','Disbursed','Front Office','Back Office','Staff Strength','Fed Funds','MY GDP','US GDP','FX','USD Index','Bursa','OPR']]
depen = v4.iloc[np.where(~v4['YoY Growth'].isna())][['YoY Growth','YoY App',
                                                     'YoY Dis','Front Office','Back Office',
                                                     'YoY Staff','Fed Funds','MY GDP','US GDP',
                                                     'FX','USD Index','Bursa','OPR']]

plt.figure(figsize=(10,6))
sns.heatmap(indepen.corr(),annot=True,square=True)
plt.title('Correlation', size=18);
plt.xticks(size=13)
plt.yticks(size=13)
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(depen.corr(),annot=True,square=True)
plt.title('Correlation', size=18);
plt.xticks(size=13)
plt.yticks(size=13)
plt.show()

sns.pairplot(depen,
                 markers="+",
                diag_kind='kde',
                kind='reg',
                plot_kws={'line_kws':{'color':'#aec6cf'},'scatter_kws':{'alpha':0.7,'color':'red'}},corner=True);

#df_ori.head(1)

#model

#Buang application & bursa on first run

Xs = depen[['YoY Growth','YoY App','YoY Dis','YoY Staff','Fed Funds','MY GDP','US GDP','FX','USD Index','Bursa','OPR']]
ys = depen["YoY Growth"]

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


#Check Stationary
ikan = "YoY Growth"
X = depen[ikan].values
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

#Statinary is Bursa, MY GDP, Fed Funds

#Differencing
    
depen['YoY Growth (DF)'] = (depen['YoY Growth'] - depen['YoY Growth'].shift(1)).fillna(0)
depen['YoY Growth (DF2)'] = (depen['YoY Growth (DF)'] - depen['YoY Growth (DF)'].shift(1)).fillna(0)

depen['Application (DF)'] = (depen['Application'] - depen['Application'].shift(1)).fillna(0)
depen['Application (DF2)'] = (depen['Application (DF)'] - depen['Application (DF)'].shift(1)).fillna(0)

depen['Disbursed (DF)'] = (depen['Disbursed'] - depen['Disbursed'].shift(1)).fillna(0)
depen['Disbursed (DF2)'] = (depen['Disbursed (DF)'] - depen['Disbursed (DF)'].shift(1)).fillna(0)
depen['Disbursed (DF3)'] = (depen['Disbursed (DF2)'] - depen['Disbursed (DF2)'].shift(1)).fillna(0)
depen['Disbursed (DF4)'] = (depen['Disbursed (DF3)'] - depen['Disbursed (DF3)'].shift(1)).fillna(0)
depen['Disbursed (DF5)'] = (depen['Disbursed (DF4)'] - depen['Disbursed (DF4)'].shift(1)).fillna(0)



























































#---------------------------------------Internal---------------------------------------------------------------------------------------------
df_i = Internal[["Position as At","Total Outstanding Amount",
                 "Application","Disbursed","Front Office",
                 "Back Office","Staff Strength"]]

df_i['YoY Growth'] = df_i["Total Outstanding Amount"].pct_change(periods=1) * 100
df_i['YoY App'] = df_i["Application"].pct_change(periods=1) * 100
df_i['YoY Dis'] = df_i["Disbursed"].pct_change(periods=1) * 100
df_i['YoY FO'] = df_i["Front Office"].pct_change(periods=1) * 100
df_i['YoY BO'] = df_i["Back Office"].pct_change(periods=1) * 100
df_i['YoY Staff'] = df_i["Staff Strength"].pct_change(periods=1) * 100

df_i_corr = df_i.iloc[np.where((~df_i["YoY Staff"].isna())&
                               (df_i["YoY Staff"])>0)][["YoY Growth",
                                                            "Application","Disbursed",
                                                            "Front Office","Back Office",
                                                            "Staff Strength"]]

#df_i["Position as At"].value_counts()

sns.pairplot(df_i_corr,
                 markers="+",
                diag_kind='kde',
                kind='reg',
                plot_kws={'line_kws':{'color':'#aec6cf'},'scatter_kws':{'alpha':0.7,'color':'red'}},corner=True);

plt.figure(figsize=(10,6))
sns.heatmap(df_i_corr.corr(),annot=True,square=True)
plt.title('Correlation Internal', size=18);
plt.xticks(size=13)
plt.yticks(size=13)
plt.show()

#model
Xs = df_i_corr[["YoY NT","YoY T","YoY Pending","YoY Disbrused","YoY Cancelled","YoY Write Off","YoY Settled"]]
ys = df_i_corr["YoY Growth"]

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
intercept = +9.24984479885388
x1 = -18.64739986*df_i_corr["YoY NT"]
x2 = -12.1191211*df_i_corr["YoY T"]
x3 = +0.14217088*df_i_corr["YoY Pending"]
x4 = +30.52923365*df_i_corr["YoY Disbrused"]
x5 = -0.76370631*df_i_corr["YoY Cancelled"]
x6 = +0.08453656*df_i_corr["YoY Write Off"]
x7 = -1.10638601*df_i_corr["YoY Settled"]

df_i_corr["YoY Growth_P"] = intercept + x1 + x2 + x3 + x4 + x5 + x6 + x7

df_i_corr[["YoY Growth","YoY Growth_P"]]

#Graph
plt.figure(figsize=(10,4))
plt.plot(df_i_corr["YoY Growth"], color='b')
plt.plot(df_i_corr["YoY Growth_P"], color='r')
plt.legend(['Growth','Predicted'], fontsize=8)
plt.show()

#Prediction

#example nk 15% growth 5400706.48015064
#df_i_corr.tail(1)

#as of Latest Q1 2024
NT = 5.88
T = 10.34
Pending = -3.77
Disbrused = 7.5
Cancelled = 3.19
WriteOff = 0
Settled = -0.99

intercept = +9.24984479885388
x1 = -18.64739986*NT
x2 = -12.1191211*T
x3 = +0.14217088*Pending
x4 = +30.52923365*Disbrused
x5 = -0.76370631*Cancelled
x6 = +0.08453656*WriteOff
x7 = 1.10638601*Settled

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
#ADF
#"YoY Growth","YoY NT","YoY T","YoY Pending","YoY Disbrused","YoY Cancelled",
#"YoY Write Off","YoY Settled"

#Check Stationary
ikan = "YoY T (DF2)"
X = df_i_corr[ikan].values
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
df_i_corr['YoY NT (DF)'] = (df_i_corr['YoY NT'] - df_i_corr['YoY NT'].shift(1)).fillna(0)
df_i_corr['YoY T (DF)'] = (df_i_corr['YoY T'] - df_i_corr['YoY T'].shift(1)).fillna(0)
df_i_corr['YoY Pending (DF)'] = (df_i_corr['YoY Pending'] - df_i_corr['YoY Pending'].shift(1)).fillna(0)
df_i_corr['YoY Cancelled (DF)'] = (df_i_corr['YoY Cancelled'] - df_i_corr['YoY Cancelled'].shift(1)).fillna(0)
df_i_corr['YoY Write Off (DF)'] = (df_i_corr['YoY Write Off'] - df_i_corr['YoY Write Off'].shift(1)).fillna(0)
df_i_corr['YoY Disbrused (DF)'] = (df_i_corr['YoY Disbrused'] - df_i_corr['YoY Disbrused'].shift(1)).fillna(0)
df_i_corr['YoY Settled (DF)'] = (df_i_corr['YoY Settled'] - df_i_corr['YoY Settled'].shift(1)).fillna(0)
#df_i_corr['YoY NT (DF)'].dropna().plot()

#2nd Differencing
df_i_corr['YoY NT (DF2)'] = (df_i_corr['YoY NT (DF)'] - df_i_corr['YoY NT (DF)'].shift(1)).fillna(0)
df_i_corr['YoY T (DF2)'] = (df_i_corr['YoY T (DF)'] - df_i_corr['YoY T (DF)'].shift(1)).fillna(0)

#Transformation
#df_i_corr['YoY NT (DF)'] = np.log(df_i_corr['YoY NT'])
#df_i_corr['YoY NT (DF)']  = (df_i_corr['YoY NT (DF)']  - df_i_corr['YoY NT (DF)'] .shift(1)).fillna(0)
#df_i_corr['YoY NT (DF)'] .dropna().plot()

























































#-------------------------------------External---------------------------------------
df_e = External[["Position as At","Total Outstanding Amount","Fed Funds","MY GDP",
                 "US GDP","FX","USD Index","Bursa","OPR"]]

df_e['YoY Growth'] = df_e["Total Outstanding Amount"].pct_change(periods=1) * 100

df_e_corr = df_e.iloc[np.where((~df_e["YoY Growth"].isna())&(df_e["Position as At"]!='2024 Qtr1'))][["YoY Growth","Fed Funds","MY GDP","US GDP","FX","USD Index","Bursa","OPR"]]


sns.pairplot(df_e_corr,
                 markers="+",
                diag_kind='kde',
                kind='reg',
                plot_kws={'line_kws':{'color':'#aec6cf'},'scatter_kws':{'alpha':0.7,'color':'red'}},corner=True);

plt.figure(figsize=(10,6))
sns.heatmap(df_e_corr.corr(),annot=True,square=True)
plt.title('Correlation External', size=18);
plt.xticks(size=13)
plt.yticks(size=13)
plt.show()

X = df_e_corr[["Fed Funds","MY GDP","US GDP (DF2)","FX (DF)","MY10 Gov Bond Yield (DF2)","USD Index (DF)","Bursa","Brent Oil Price"]]
y = df_e_corr["YoY Growth"]

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.intercept_)
print(regr.coef_)
print(regr.score(X, y))

import statsmodels.api as sm


#add constant to predictor variables
x = sm.add_constant(X)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())


#ADF - Stationary Test

#YoY Growth","Fed Funds","MY GDP","US GDP","FX",
#"MY10 Gov Bond Yield","USD Index","Bursa","Brent Oil Price"

#External
ayam = "Brent Oil Price"
X = df_e_corr[ayam].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

if result[0] < result[4]["5%"]:
    print (ayam+" Reject Ho - Time Series is Stationary")
else:
    print (ayam+" Failed to Reject Ho - Time Series is Non-Stationary")


df_e_corr['US GDP (DF)'] = (df_e_corr['US GDP'] - df_e_corr['US GDP'].shift(1)).fillna(0)
df_e_corr['US GDP (DF2)'] = (df_e_corr['US GDP (DF)'] - df_e_corr['US GDP (DF)'].shift(1)).fillna(0)

df_e_corr['FX (DF)'] = (df_e_corr['FX'] - df_e_corr['FX'].shift(1)).fillna(0)

df_e_corr['MY10 Gov Bond Yield (DF)'] = (df_e_corr['MY10 Gov Bond Yield'] - df_e_corr['MY10 Gov Bond Yield'].shift(1)).fillna(0)
df_e_corr['MY10 Gov Bond Yield (DF2)'] = (df_e_corr['MY10 Gov Bond Yield (DF)'] - df_e_corr['MY10 Gov Bond Yield (DF)'].shift(1)).fillna(0)

df_e_corr['USD Index (DF)'] = (df_e_corr['USD Index'] - df_e_corr['USD Index'].shift(1)).fillna(0)



#Prediction

#example nk 15% growth 5400706.48015064
#df_i_corr.tail(1)

#as of Latest Q1 2024
Fed = 5.25
MY = 3.4
US = -0.3
FX = 7.17e-02
MY10 = -0.06
USD = 1.27e+00
Bursa = 1447.04
Brent = 76.64

intercept = +156.1288
x1 = -5.6740*Fed
x2 = -0.9428*MY
x3 = +0.9725*US
x4 = +124.6216*FX
x5 = -26.5592*MY10
x6 = -2.4927*USD
x7 = -0.1177*Bursa
x8 = +0.4639*Brent

fifteen = 10

newNT = (fifteen - (intercept+x2+x3+x4+x5+x6+x7+x8))/(x1/Fed)
print(newNT)

newT = (fifteen - (intercept+x1+x3+x4+x5+x6+x7+x8))/(x2/MY)
print(newT)

newP = (fifteen - (intercept+x1+x2+x4+x5+x6+x7+x8))/(x3/US)
print(newP)

newD = (fifteen - (intercept+x1+x2+x3+x5+x6+x7+x8))/(x4/FX)
print(newD)

newC= (fifteen - (intercept+x1+x2+x3+x4+x6+x7+x8))/(x5/MY10)
print(newC)

newW= (fifteen - (intercept+x1+x2+x3+x4+x5+x7+x8))/(x6/USD)
print(newW)

newS= (fifteen - (intercept+x1+x2+x3+x4+x5+x6+x8))/(x7/Bursa)
print(newS)

newB= (fifteen - (intercept+x1+x2+x3+x4+x5+x6+x7))/(x8/Brent)
print(newB)

#auto correlation (Durbin Watson Test)

#Apply Differencing for non stationary data to become stationary  


#regr.coef_

#predictedCO2 = regr.predict([[2300, 1300]])

###############################################
#bar
#import matplotlib.pyplot as plt
#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["figure.autolayout"] = True
##df = pd.DataFrame(dict(data=[2, 4, 1, 5, 9, 6, 0, 7]))
#fig, ax = plt.subplots()
#df_ori['Total Outstanding Amount'].plot(kind='bar', color='red')
#df_ori['YoY Growth'].plot(kind='line', marker='*', color='black', ms=100)
#plt.show()

###############################################
#bar
#df_ori["Total Outstanding Amount"].describe()
#df_ori["YoY Growth"].describe()
#fig = plt.figure(figsize=(10,7))
#plt.boxplot(df_ori["Total Outstanding Amount"])
#plt.show()