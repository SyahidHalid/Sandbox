import pandas as pd #data manipulation and analysis
import numpy as np #linear Algebra
import matplotlib.pyplot as plt #data visualisation
import seaborn as sns # data visualisation
import warnings #ignore warnings
import openpyxl
warnings.filterwarnings('ignore')

pd.set_option("display.max_columns", None) 
pd.set_option("display.max_colwidth", 1000) #huruf dlm column
pd.set_option("display.max_rows", 100)
pd.set_option("display.precision", 2) #2 titik perpuluhan

df_ori = pd.read_excel(r"C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Analytics\Loan Growth Model\2024\Loan Growth - Working.xlsx", sheet_name='External')

df_ori

#df = df_ori.iloc[np.where(df_ori.Year>=2020)][["LAF (RM'000)",
#                                               'Fed Interest Rates (Actual)',
#                                               'FX (1 USD)',
#                                               'OPR (BANK NEGARA)',
#                                               'KLSE Index @ FTSE Bursa MY KLCI Index @ Malaysia Stock Market',
#                                               'GDP per capita growth (annual %) - Malaysia']]

df_ori.head()

df_ori.info()

df_ori.shape

df_ori.columns

df_ori.describe() #for non object

#df_ori.describe(include='O') #for object

#df_ori.['column'].unique() #tuk object value unique

#df_ori.isnull().sum() #for unique

#df_ori[df_ori.duplicated]

#df_ori.drop_duplicates(keep='first', inplace=True)

plt.figure(figsize=(10,6))
sns.distplot(df_ori["LAF (000)"],color='r')
plt.title('Gross Financing Asset', size=18)
plt.xlabel('LAF', size=14)
plt.ylabel('Density', size=14)
plt.show()

#to show distribution and kernel density,
#below is positively skew between 0.3 to 0.6