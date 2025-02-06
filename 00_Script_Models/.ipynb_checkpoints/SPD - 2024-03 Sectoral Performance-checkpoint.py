file1 = "202112 - 202012"
file2 = "202212 - 202201" 
file3 = "202312 - 202301" 
file4 = "202412 - 202401"

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None) 
pd.set_option("display.max_colwidth", 1000) #huruf dlm column
pd.set_option("display.max_rows", 100)
pd.set_option("display.precision", 2) #2 titik perpuluhan

loc1 = r"C:\\Users\\syahidhalid\\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\\Exim _ Syahid\\Analytics\\Loan Database\\LBD "+str(file1)+" (CSP Sector).xlsx"
loc2 = r"C:\\Users\\syahidhalid\\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\\Exim _ Syahid\\Analytics\\Loan Database\\LBD "+str(file2)+" (CSP Sector).xlsx"
loc3 = r"C:\\Users\\syahidhalid\\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\\Exim _ Syahid\\Analytics\\Loan Database\\LBD "+str(file3)+" (CSP Sector).xlsx"
loc4 = r"C:\\Users\\syahidhalid\\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\\Exim _ Syahid\\Analytics\\Loan Database\\LBD "+str(file4)+" (CSP Sector).xlsx"

#f1 = pd.read_excel(loc1, header = 0, sheet_name="Notes")
#f2 = pd.read_excel(loc2, header = 0, sheet_name="Notes")
#f3 = pd.read_excel(loc3, header = 0, sheet_name="Notes")
#f4 = pd.read_excel(loc4, header = 0, sheet_name="Notes")

#frames = [f1, f2, f3, f4] 
#a = pd.concat(frames)

#a.iloc[np.where(a["Yes/No"]=="Yes")]
#================================================================================


sc2012 = pd.read_excel(loc1, header = 0, sheet_name="Dec20")
sc2103 = pd.read_excel(loc1, header = 0, sheet_name="Mar21")
sc2106 = pd.read_excel(loc1, header = 0, sheet_name="Jun21")
sc2109 = pd.read_excel(loc1, header = 0, sheet_name="Sep21")
sc2112 = pd.read_excel(loc1, header = 0, sheet_name="Dec21")
sc2203 = pd.read_excel(loc2, header = 0, sheet_name="Mar22")
sc2206 = pd.read_excel(loc2, header = 0, sheet_name="Jun22")
sc2209 = pd.read_excel(loc2, header = 0, sheet_name="Sep22")
sc2212 = pd.read_excel(loc2, header = 0, sheet_name="Dec22")
sc2303 = pd.read_excel(loc3, header = 0, sheet_name="Mar23")
sc2306 = pd.read_excel(loc3, header = 0, sheet_name="Jun23")
sc2309 = pd.read_excel(loc3, header = 0, sheet_name="Sep23")
sc2312 = pd.read_excel(loc3, header = 0, sheet_name="Dec23")
sc2402 = pd.read_excel(loc4, header = 0, sheet_name="Feb24")


filt1 = sc2012.iloc[np.where(sc2012['Status'].isin(['Impaired','Active-Watchlist','Active','Active-Watchlist/Overdue']))]
filt2 = sc2103.iloc[np.where(sc2103['Status'].isin(['Active-Overdue','Impaired','Active-Watchlist','Active','Active-Watchlist/Overdue']))]
filt3 = sc2106.iloc[np.where(sc2106['Status'].isin(['Impaired','Active-Watchlist','Active','Active-Watchlist/Overdue']))]
filt4 = sc2109.iloc[np.where(sc2109['Status'].isin(['Impaired','Active-Watchlist','Active','Active-Watchlist/Overdue']))]
filt5 = sc2112.iloc[np.where(sc2112['Status'].isin(['Impaired','Active-Watchlist','Active','Active-Watchlist/Overdue']))]
filt6 = sc2203.iloc[np.where(sc2203['Status'].isin(['Impaired','Active-Watchlist','Active','Active-Watchlist/Overdue']))]
filt7 = sc2206.iloc[np.where(sc2206['Status'].isin(['Impaired','Active-Watchlist','Active','Active-Watchlist/Overdue']))]
filt8 = sc2209.iloc[np.where(sc2209['Status'].isin(['Impaired','Active-Watchlist','Active','Active-Watchlist/Overdue']))]
filt9 = sc2212.iloc[np.where(sc2212['Status'].isin(['Impaired','Active-Watchlist','Active','Active-Watchlist/Overdue']))]
filt10 = sc2303.iloc[np.where(sc2303['Status'].isin(['Impaired','Active-Watchlist','Active','Active-Watchlist/Overdue']))]
filt11 = sc2306.iloc[np.where(sc2306['Status'].isin(['Impaired','Active-Watchlist','Active','Active-Watchlist/Overdue']))]
filt12 = sc2309.iloc[np.where(sc2309['Status'].isin(['Impaired','Active-Watchlist','Active','Active-Watchlist/Overdue']))]
filt13 = sc2312.iloc[np.where(sc2312['Status'].isin(['Active-Overdue','Impaired','Active-Watchlist','Active','Active-Watchlist/Overdue']))]
filt14 = sc2402.iloc[np.where(sc2402['Status'].isin(['Active-Overdue','Impaired','Active-Watchlist','Active','Active-Watchlist/Overdue']))]

#validation
filt1['Status'].value_counts()
sc2012['Status'].value_counts()

filt2['Status'].value_counts()
sc2103['Status'].value_counts()

filt3['Status'].value_counts()
sc2106['Status'].value_counts()

filt4['Status'].value_counts()
sc2109['Status'].value_counts()

filt5['Status'].value_counts()
sc2112['Status'].value_counts()

filt6['Status'].value_counts()
sc2203['Status'].value_counts()

filt7['Status'].value_counts()
sc2206['Status'].value_counts()

filt8['Status'].value_counts()
sc2209['Status'].value_counts()

filt9['Status'].value_counts()
sc2212['Status'].value_counts()

filt10['Status'].value_counts()
sc2303['Status'].value_counts()

filt11['Status'].value_counts()
sc2306['Status'].value_counts()

filt12['Status'].value_counts()
sc2309['Status'].value_counts()

filt13['Status'].value_counts()
sc2312['Status'].value_counts()

filt14['Status'].value_counts()
sc2402['Status'].value_counts()

#df.groupby(['group']).agg(['min', 'max', 'count', 'nunique'])

#filt1.head(1) tuk cari outstanding
filt1g = filt1.fillna(0).groupby(['BNM Main Sector']).\
agg({'CIF Number': 'nunique',\
     'EXIM Account No.': 'nunique',\
     'Outstanding Amount (RM)':'sum'}).reset_index()

#filt2.head(1)
filt2g = filt2.fillna(0).groupby(['BNM Main Sector']).\
agg({'CIF Number': 'nunique',\
     'EXIM Account No.': 'nunique',\
     'Total Banking Exposure (MYR)':'sum'}).reset_index()

filt3g = filt3.fillna(0).groupby(['BNM Main Sector']).\
agg({'CIF Number': 'nunique',\
     'EXIM Account No.': 'nunique',\
     'Total Banking Exposure (MYR)':'sum'}).reset_index()

filt4g = filt4.fillna(0).groupby(['BNM Main Sector']).\
agg({'CIF Number': 'nunique',\
     'EXIM Account No.': 'nunique',\
     'Total Loans Outstanding (MYR)':'sum'}).reset_index()

filt5g = filt5.fillna(0).groupby(['BNM Main Sector']).\
agg({'CIF Number': 'nunique',\
     'EXIM Account No.': 'nunique',\
     'Total Loans Outstanding (MYR)':'sum'}).reset_index()

filt6g = filt6.fillna(0).groupby(['BNM Main Sector']).\
agg({'CIF Number': 'nunique',\
     'EXIM Account No.': 'nunique',\
     'Total Loans Outstanding (MYR)':'sum'}).reset_index()

filt7g = filt7.fillna(0).groupby(['BNM Main Sector']).\
agg({'CIF Number': 'nunique',\
     'EXIM Account No.': 'nunique',\
     'Total Loans Outstanding (MYR)':'sum'}).reset_index()

filt8g = filt8.fillna(0).groupby(['BNM Main Sector']).\
agg({'CIF Number': 'nunique',\
     'EXIM Account No.': 'nunique',\
     'Total Loans Outstanding (MYR)':'sum'}).reset_index()

filt9g = filt9.fillna(0).groupby(['BNM Main Sector']).\
agg({'CIF Number': 'nunique',\
     'EXIM Account No.': 'nunique',\
     'Total Loans Outstanding (MYR)':'sum'}).reset_index()

filt10g = filt10.fillna(0).groupby(['BNM Main Sector']).\
agg({'CIF Number': 'nunique',\
     'EXIM Account No.': 'nunique',\
     'Total Loans Outstanding (MYR)':'sum'}).reset_index()

filt11g = filt11.fillna(0).groupby(['BNM Main Sector']).\
agg({'CIF Number': 'nunique',\
     'EXIM Account No.': 'nunique',\
     'Total Loans Outstanding (MYR)':'sum'}).reset_index()

filt12g = filt12.fillna(0).groupby(['BNM Main Sector']).\
agg({'CIF Number': 'nunique',\
     'EXIM Account No.': 'nunique',\
     'Total Loans Outstanding (MYR)':'sum'}).reset_index()

filt13g = filt13.fillna(0).groupby(['BNM Main Sector']).\
agg({'CIF Number': 'nunique',\
     'EXIM Account No.': 'nunique',\
     'Total Loans Outstanding (MYR)':'sum'}).reset_index()

filt14g = filt14.fillna(0).groupby(['BNM Main Sector']).\
agg({'CIF Number': 'nunique',\
     'EXIM Account No.': 'nunique',\
     'Total Loans Outstanding (MYR)':'sum'}).reset_index()

#validation
#sum(test['EXIM Account No.'])
#filt1.shape

combine = filt1g.merge(filt2g, on = 'BNM Main Sector', how = 'outer', suffixes=('_1','_2')).\
merge(filt3g, on = 'BNM Main Sector', how = 'outer', suffixes=('','_3')).\
merge(filt4g, on = 'BNM Main Sector', how = 'outer', suffixes=('','_4')).\
merge(filt5g, on = 'BNM Main Sector', how = 'outer', suffixes=('','_5')).\
merge(filt6g, on = 'BNM Main Sector', how = 'outer', suffixes=('','_6')).\
merge(filt7g, on = 'BNM Main Sector', how = 'outer', suffixes=('','_7')).\
merge(filt8g, on = 'BNM Main Sector', how = 'outer', suffixes=('','_8')).\
merge(filt9g, on = 'BNM Main Sector', how = 'outer', suffixes=('','_9')).\
merge(filt10g, on = 'BNM Main Sector', how = 'outer', suffixes=('','_10')).\
merge(filt11g, on = 'BNM Main Sector', how = 'outer', suffixes=('','_11')).\
merge(filt12g, on = 'BNM Main Sector', how = 'outer', suffixes=('','_12')).\
merge(filt13g, on = 'BNM Main Sector', how = 'outer', suffixes=('','_13')).\
merge(filt14g, on = 'BNM Main Sector', how = 'outer', suffixes=('','_14')).fillna(0)

combine.to_excel(r'C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Analytics\Sectoral and Geographical\CSP\Sector Python - Working.xlsx', index = False)

#writer = pd.ExcelWriter(r"T:\MIB Risk Management\Credit Risk Analytics - Historical\02_Data_Source\20_CRP\6.0 Impaired Loan\\"+str(date)[:4]+"\\"+str(date)[:6]+"\working\SUMM_IF_"+str(date)[:6]+"(python).xlsx", engine='xlsxwriter')
#SUMM_IF.to_excel(writer, sheet_name='SUMM', index = False)
#LIST_IF_PDT.to_excel(writer, sheet_name='LIST_IF_PDT', index = False)
#writer.save()

#cust.columns = cust.columns.str.replace(" ", "_")
#cust.M_SUB_SUB_MARKET_SEGMENT.replace({'    ': -9999}, inplace=True
#.rename(columns={'a':'b'})