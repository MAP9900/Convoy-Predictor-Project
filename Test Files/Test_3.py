
import pandas as pd
data = pd.read_csv('Excel_Files/Complete_Data/SC_Data.csv')


import matplotlib.pyplot as plt

# Convert date columns to datetime format
data['Depart_Date'] = pd.to_datetime(data['Depart_Date'], errors='coerce')
data['Arrive_Date'] = pd.to_datetime(data['Arrive_Date'], errors='coerce')

# Sort data by Depart_Date for trend analysis
data_sorted = data.sort_values(by='Depart_Date')

# Plot trends over time
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

# Number of Ships, Escort Ships, and Total Tons Over Time
axes[0].plot(data_sorted['Depart_Date'], data_sorted['Number of Ships'], label='Number of Ships')
axes[0].plot(data_sorted['Depart_Date'], data_sorted['Number of Escort Ships'], label='Number of Escort Ships')
axes[0].set_ylabel('Number of Ships')
axes[0].set_title('Number of Ships and Escort Ships Over Time')
axes[0].legend()

# Number of Ships Sunk and Total Tons of Ships Sunk Over Time
axes[1].plot(data_sorted['Depart_Date'], data_sorted['Number of Ships Sunk'], label='Number of Ships Sunk')
axes[1].set_ylabel('Number of Ships Sunk')
axes[1].set_title('Number of Ships Sunk Over Time')
axes[1].legend()

# Total Tons of Convoy and Total Tons of Ships Sunk Over Time
axes[2].plot(data_sorted['Depart_Date'], data_sorted['Total Tons of Convoy'], label='Total Tons of Convoy')
axes[2].plot(data_sorted['Depart_Date'], data_sorted['Total Tons of Ships Sunk'], label='Total Tons of Ships Sunk')
axes[2].set_ylabel('Total Tons')
axes[2].set_title('Total Tons of Convoy and Ships Sunk Over Time')
axes[2].legend()

plt.tight_layout()
plt.show()


from scipy.stats import pearsonr

# Correlation between Number of Escort Ships and Number of Ships Sunk
correlation_escort_sunk, p_value_escort_sunk = pearsonr(data['Number of Escort Ships'], data['Number of Ships Sunk'])

# Correlation between Total Tons of Convoy and Number of Ships Sunk
correlation_tons_sunk, p_value_tons_sunk = pearsonr(data['Total Tons of Convoy'], data['Number of Ships Sunk'])

correlation_escort_sunk, p_value_escort_sunk, correlation_tons_sunk, p_value_tons_sunk



# Divide number of ships into ranges
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
data['Ship Range'] = pd.cut(data['Number of Ships'], bins=bins, labels=labels, right=False)

# Group data by ship range
grouped_ships = data.groupby('Ship Range')

# Initialize lists to store results
ships_sunk_by_ship_range = []

# Loop through each group and calculate the total number of ships sunk
for name, group in grouped_ships:
    total_ships_sunk = group['Number of Ships Sunk'].sum()
    ships_sunk_by_ship_range.append((name, total_ships_sunk))

# Plot results
fig, ax = plt.subplots(figsize=(10, 5))

# Number of Ships Sunk by Ship Range
ax.bar([str(label) for label, _ in ships_sunk_by_ship_range], [ships_sunk for _, ships_sunk in ships_sunk_by_ship_range])
ax.set_ylabel('Number of Ships Sunk')
ax.set_title('Number of Ships Sunk by Ship Range')

plt.tight_layout()
plt.show()






# Divide data into different time periods
data['Year'] = data['Depart_Date'].dt.year
time_periods = data['Year'].unique()

# Initialize lists to store results
ships_sunk_by_year = []
tons_sunk_by_year = []

# Loop through each time period and calculate the total number of ships sunk and total tons of ships sunk
for year in time_periods:
    if pd.isna(year):
        continue  # Skip NaN values
    year_data = data[data['Year'] == year]
    total_ships_sunk = year_data['Number of Ships Sunk'].sum()
    total_tons_sunk = year_data['Total Tons of Ships Sunk'].sum()
    ships_sunk_by_year.append((year, total_ships_sunk))
    tons_sunk_by_year.append((year, total_tons_sunk))

# Sort results by year
ships_sunk_by_year.sort(key=lambda x: x[0])
tons_sunk_by_year.sort(key=lambda x: x[0])

# Plot results
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

# Number of Ships Sunk by Year
axes[0].bar([str(int(year)) for year, _ in ships_sunk_by_year], [ships_sunk for _, ships_sunk in ships_sunk_by_year])
axes[0].set_ylabel('Number of Ships Sunk')
axes[0].set_title('Number of Ships Sunk by Year')

# Total Tons of Ships Sunk by Year
axes[1].bar([str(int(year)) for year, _ in tons_sunk_by_year], [tons_sunk for _, tons_sunk in tons_sunk_by_year])
axes[1].set_ylabel('Total Tons of Ships Sunk')
axes[1].set_title('Total Tons of Ships Sunk by Year')

plt.tight_layout()
plt.show()


# Group data by number of escort ships
grouped_escort = data.groupby('Number of Escort Ships')

# Initialize lists to store results
ships_sunk_by_escort = []
tons_sunk_by_escort = []

# Loop through each group and calculate the total number of ships sunk and total tons of ships sunk
for name, group in grouped_escort:
    total_ships_sunk = group['Number of Ships Sunk'].sum()
    total_tons_sunk = group['Total Tons of Ships Sunk'].sum()
    ships_sunk_by_escort.append((name, total_ships_sunk))
    tons_sunk_by_escort.append((name, total_tons_sunk))

# Sort results by number of escort ships
ships_sunk_by_escort.sort(key=lambda x: x[0])
tons_sunk_by_escort.sort(key=lambda x: x[0])

# Plot results
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

# Number of Ships Sunk by Number of Escort Ships
axes[0].bar([str(int(escort)) for escort, _ in ships_sunk_by_escort], [ships_sunk for _, ships_sunk in ships_sunk_by_escort])
axes[0].set_ylabel('Number of Ships Sunk')
axes[0].set_title('Number of Ships Sunk by Number of Escort Ships')

# Total Tons of Ships Sunk by Number of Escort Ships
axes[1].bar([str(int(escort)) for escort, _ in tons_sunk_by_escort], [tons_sunk for _, tons_sunk in tons_sunk_by_escort])
axes[1].set_ylabel('Total Tons of Ships Sunk')
axes[1].set_title('Total Tons of Ships Sunk by Number of Escort Ships')

plt.tight_layout()
plt.show()
