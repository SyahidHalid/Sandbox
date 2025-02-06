#https://medium.com/@tirthamutha/time-series-forecasting-using-sarima-in-python-8b75cd3366f2

#input dataset

#Plot
sns.lineplot(x='date',y='meantemp',data=data)


#Autocorrelation, Seasonality and Stationarity
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(data['meantemp'])
#From the above graphs it is demonstrated that there is seasonality in the dataset. A damped sinusoidal wave is seen with a trough around every 350–400 lags.


from statsmodels.graphics.tsaplots import plot_acf


plot_acf(data['meantemp'])
plt.show()

#ACF plots show the correlation between a time series and a lagged version of itself. The ACF graph is decreasing exponentially.



from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(data['meantemp'])
plt.show()
#PACF plots show the correlation between a time series and a lagged version of itself after removing the effects of intervening observations. The PACF graph is abrupt.


decomposition=seasonal_decompose(data['meantemp'],model='additive',period=12)
decomposition.plot()
plt.show()
#Time series decomposition is a statistical technique that involves breaking down a time series into its underlying components to better understand its patterns and improve forecasting accuracy. The most common components of a time series are trend, seasonality, and remainder.



#ADF
dftest = adfuller(data.meantemp, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)

#ARIMA
data_diff=data['meantemp'].diff(periods=350)

dftest = adfuller(data_diff.dropna(), autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)


#Peforming the seasonal ARIMA
    import pmdarima as pmd

model=pmd.auto_arima(data['meantemp'],start_p=1,start_q=1,test='adf',m=12,seasonal=True,trace=True)
#Best model: ARIMA(1,1,1)(1,0,1)[12]


sarima=SARIMAX(data['meantemp'],order=(1,1,1),seasonal_order=(1,0,1,12))
predicted=sarima.fit().predict();predicted

then buat
Actual vs. Predicted graph of meantemp (training data)

#Actual vs. Predicted graph of meantemp (test data)


