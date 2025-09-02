
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
import matplotlib.dates as mdates
sc = pd.read_csv('Excel_Files/Complete_Data/SC_Data.csv')
#print(sc.head())
sc.head()
def Divide(a,b):
    if b == 0:
        return np.nan
    return a / b
sc['Overall Sink Rate'] = sc.apply(lambda row: Divide(row['Number of Ships Sunk'], row['Number of Ships'], ), axis=1)
sc['Escort Sink Rate'] = sc.apply(lambda row: Divide(row['Number of Escorts Sunk'], row['Number of Escort Ships']), axis=1)
sc['Straggler Sink Rate'] = sc.apply(lambda row: Divide(row['Number of Stragglers Sunk'], row['Number of Stragglers'],), axis=1)
sc['Depart_Date'] = pd.to_datetime(sc['Depart_Date'], errors='coerce')
sc = sc.fillna(0)
sc.head()

window = 5
sc['Moving_Avg_Ships'] = sc['Number of Ships'].rolling(window=window).mean()
fig, ax = plt.subplots(figsize=(10, 5), facecolor='lightgrey')
ax.bar(sc['Depart_Date'], sc['Number of Ships'], color='steelblue')
ax.set_xlabel('Departure Date')
ax.set_ylabel('Total Number of Ships')
ax.set_facecolor('lightgrey')
US_War = pd.to_datetime('1941-12-07')
VE_Day = pd.to_datetime('1945-05-05')
plt.axvline(x=US_War, color='black', linestyle='--', linewidth=1, label='US Enters the War')
plt.axvline(x=VE_Day, color='black', linestyle='--', linewidth=1, label='VE Day')
plt.title('Size of SC Convoys (in ships) Over Time')
#ax2 = ax.twinx()
ax.plot(sc['Depart_Date'], sc['Moving_Avg_Ships'], color='red', label='Moving Average')
#x = mdates.date2num(sorted['Depart_Date'])
#y = sorted['Number of Ships']
#z = np.polyfit(x, sorted['Number of Ships'], 25)
#p = np.poly1d(z)
#plt.plot(x, p(x))
plt.yticks(np.arange(0,110,10))
plt.bar(sc['Depart_Date'], sc['Number of Escort Ships'], color='black', label='Number of Escorts')
plt.legend(title='Legend', facecolor='lightgrey', loc='upper left', markerscale=.5, fontsize=8)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
ax.xaxis.set_major_locator(MonthLocator(bymonth=(1, 7)))
plt.show()