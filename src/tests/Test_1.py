from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler as SS
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV


# Load the provided dataset
file_path = 'heart_disease.csv'
heart_disease_data = pd.read_csv(file_path, delimiter=';')

# Display the first few rows of the dataset to understand its structure
heart_disease_data.head()
SS = SS()

# Selecting the required features and target
X = np.array(heart_disease_data[['Age', 'Resting_Blood_Pressure', 'Cholesterol', 'Max Heart Rate', 'ST_depression']])
y = heart_disease_data['site'].values

# Setting the parameters for cross-validation
k_values = range(10, 41)
random_state = 146
cv_folds = 5

#K-Fold Cross Validation Function
def K_Fold(model, X, y, K, scaler=None, random_state=1945):
    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)
    train_scores = []
    test_scores = []
    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain]
        Xtest = X[idxTest]
        ytrain = y[idxTrain]
        ytest = y[idxTest]
        if scaler is not None:
            Xtrain = scaler.fit_transform(Xtrain)
            Xtest = scaler.transform(Xtest)
        model.fit(Xtrain, ytrain)
        train_scores.append(model.score(Xtrain, ytrain))
        test_scores.append(model.score(Xtest, ytest))
    return train_scores, test_scores

avg_tr_score = []
avg_te_score = []
for a in k_values:
    knn = KNeighborsClassifier(n_neighbors=a)
    train_scores, test_scores = K_Fold(knn, X, y, 5, SS)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))
n_neighbors = k_values[np.argmax(avg_te_score)]
print('Optimal Number of Neighbors (Test Score):', n_neighbors)


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


#^ total tons and ships over Time


from scipy.stats import pearsonr

# Correlation between Number of Escort Ships and Number of Ships Sunk
correlation_escort_sunk, p_value_escort_sunk = pearsonr(data['Number of Escort Ships'], data['Number of Ships Sunk'])

# Correlation between Total Tons of Convoy and Number of Ships Sunk
correlation_tons_sunk, p_value_tons_sunk = pearsonr(data['Total Tons of Convoy'], data['Number of Ships Sunk'])

correlation_escort_sunk, p_value_escort_sunk, correlation_tons_sunk, p_value_tons_sunk

#Stats analysis  ^



# Divide total tonnage into ranges
bins = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000]
labels = ['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k-250k', '250k-300k', '300k-350k']
data['Tonnage Range'] = pd.cut(data['Total Tons of Convoy'], bins=bins, labels=labels, right=False)

# Group data by tonnage range
grouped_tonnage = data.groupby('Tonnage Range')

# Initialize lists to store results
ships_sunk_by_tonnage = []
tons_sunk_by_tonnage = []

# Loop through each group and calculate the total number of ships sunk and total tons of ships sunk
for name, group in grouped_tonnage:
    total_ships_sunk = group['Number of Ships Sunk'].sum()
    total_tons_sunk = group['Total Tons of Ships Sunk'].sum()
    ships_sunk_by_tonnage.append((name, total_ships_sunk))
    tons_sunk_by_tonnage.append((name, total_tons_sunk))

# Plot results
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

# Number of Ships Sunk by Total Tonnage Range
axes[0].bar([str(label) for label, _ in ships_sunk_by_tonnage], [ships_sunk for _, ships_sunk in ships_sunk_by_tonnage])
axes[0].set_ylabel('Number of Ships Sunk')
axes[0].set_title('Number of Ships Sunk by Total Tonnage Range')

# Total Tons of Ships Sunk by Total Tonnage Range
axes[1].bar([str(label) for label, _ in tons_sunk_by_tonnage], [tons_sunk for _, tons_sunk in tons_sunk_by_tonnage])
axes[1].set_ylabel('Total Tons of Ships Sunk')
axes[1].set_title('Total Tons of Ships Sunk by Total Tonnage Range')

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




from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np




import pandas as pd

# Load the dataset
file_path = '/mnt/data/Complete_Convoy_Data.csv'
df = pd.read_csv(file_path)

# Apply the provided transformations
df2 = df[df['Time At Sea (Days)'] > 10]
df2 = df2.drop(columns=['Convoy Number', 'Number of Ships Sunk', 'Depart_Date', 'Arrival/Dispersal Date', 
                        'Number of Escorts Sunk', 'Number of Stragglers Sunk', 'Total Tons of Ships Sunk', 
                        'Escort Sink Percentage', 'Straggler Sink Percentage'])
df2.reset_index(drop=True, inplace=True)

# Convert 'Overall Sink Percentage' to a binary value
df2['High Risk'] = (df2['Overall Sink Percentage'] > 0).astype(int)

# Separate into features and target
X = df2.drop(columns=['Overall Sink Percentage', 'High Risk']).values
y = df2['High Risk'].values

df2.head()


# Initialize a dictionary to store the performance metrics for each year
performance_metrics = {}
# Checking the range of years in the dataset
year_range = df2['Year'].unique()
year_range.sort()
year_range


# Loop through each year from 1939 to 1945
for year in year_range:
    # Filter the data for the current year and all previous years
    df_year = df2[df2['Year'] <= year]
    
    # Separate features and target
    X_year = df_year.drop(columns=['Overall Sink Percentage', 'High Risk']).values
    y_year = df_year['High Risk'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_year, y_year, test_size=0.2, random_state=42)

    # Create and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions and calculate performance metrics
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Store the performance metrics
    performance_metrics[year] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

performance_metrics



from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Support Vector Machine Classifier
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train_class, y_train_class)
y_pred_svm = svm_classifier.predict(X_test_class)
svm_accuracy = accuracy_score(y_test_class, y_pred_svm)
svm_class_report = classification_report(y_test_class, y_pred_svm)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_class, y_train_class)
y_pred_dt = dt_classifier.predict(X_test_class)
dt_accuracy = accuracy_score(y_test_class, y_pred_dt)
dt_class_report = classification_report(y_test_class, y_pred_dt)

(svm_accuracy, svm_class_report), (dt_accuracy, dt_class_report)
