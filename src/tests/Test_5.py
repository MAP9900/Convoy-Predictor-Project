#Imports 
import pandas as pd
import numpy as np

#DataFrames

#Add average number of U-Boats operating in the Atlantic Ocean 
U_Boat = {'1939-09': (6, 2), '1939-10': (3, 4), '1939-11': (4, 2), '1939-12': (2, 1), '1940-01': (7, 2), 
               '1940-02': (8, 5), '1940-03': (3, 2), '1940-04': (24, 7), '1940-05': (5, 1), '1940-06': (12, 11), 
               '1940-07': (5, 4), '1940-08': (9, 2), '1940-09': (7, 6), '1940-10': (7, 5), '1940-11': (8, 5), 
               '1940-12': (12, 3), '1941-01': (12, 3), '1941-02': (11, 0), '1941-03': (15, 9), '1941-04': (16, 3),
               '1941-05': (16, 1), '1941-06': (25, 6), '1941-07': (20, 2), '1941-08': (35, 5), '1941-09': (36, 7),
               '1941-10': (30, 2), '1941-11': (32, 4), '1941-12': (24, 13), '1942-01': (35, 5), '1942-02': (44, 33),
               '1942-03': (45, 7), '1942-04': (42, 2), '1942-05': (49, 5), '1942-06': (61, 8), '1942-07': (73, 12),
               '1942-08': (82, 16), '1942-09': (91, 9), '1942-10': (110, 13), '1942-11': (98, 18), '1942-12': (98, 6),
               '1943-01': (105, 8), '1943-02': (106, 20), '1943-03': (114, 13), '1943-04': (110, 14), '1943-05': (116, 37),
               '1943-06': (89, 13), '1943-07': (89, 40), '1943-08': (50, 22), '1943-09': (56, 8), '1943-10': (68, 28),
               '1943-11': (66, 16), '1943-12': (58, 6), '1944-01': (57, 15), '1944-02': (54, 18), '1944-03': (52, 20),
               '1944-04': (43, 14), '1944-05': (32, 19), '1944-06': (43, 24), '1944-07': (26, 16), '1944-08': (35, 22),
               '1944-09': (47, 11), '1944-10': (21, 6), '1944-11': (23, 4), '1944-12': (30, 8), '1945-01': (30, 4),
               '1945-02': (39, 13), '1945-03': (50, 10), '1945-04': (53, 21), '1945-05': (np.nan, np.nan)}

#Convert to Data Frame
U_Boat_df = pd.DataFrame(list(U_Boat.items()), columns=['Date', 'Data'])
U_Boat_df[['Average Number in Atlantic Ocean', 'U-Boats Sunk']] = pd.DataFrame(U_Boat_df['Data'].tolist(), index=U_Boat_df.index)
U_Boat_df = U_Boat_df.drop(columns=['Data'])

U_Boat_df.to_csv('U-Boat-Data.csv')


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardizing the features before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_class)

# Applying PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_scaled)

# The number of components chosen by PCA
n_components = X_pca.shape[1]

# Transforming the test set with the same PCA and scaler
X_test_scaled = scaler.transform(X_test_class)
X_test_pca = pca.transform(X_test_scaled)

# Train a Random Forest Classifier on the transformed data
rf_classifier_pca = RandomForestClassifier(random_state=42)
rf_classifier_pca.fit(X_pca, y_train_class)
y_pred_pca = rf_classifier_pca.predict(X_test_pca)

# Evaluate the classifier after PCA transformation
pca_accuracy = accuracy_score(y_test_class, y_pred_pca)
pca_class_report = classification_report(y_test_class, y_pred_pca)

n_components, pca_accuracy, pca_class_report



from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Extracting year for time-based splitting
df2['Year'] = df2['Year'].astype(int)
years = df2['Year'].unique()

# Sorting the DataFrame by year to ensure correct time series splitting
df2.sort_values(by='Year', inplace=True)
X = df2.drop(columns=['Overall Sink Percentage', 'High Risk'])
y = df2['High Risk']

# Setting up TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=len(years) - 1) # Subtract 1 because the first year can't be used as a test

# Placeholder for storing model performance metrics
model_performance = []

# Time-based cross-validation
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Training the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    # Making predictions
    y_pred = clf.predict(X_test)

    # Computing performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Storing the results
    year = X_test['Year'].iloc[0]
    model_performance.append({'Year': year, 'Accuracy': accuracy, 'Confusion Matrix': conf_matrix, 'Classification Report': class_report})

# Display the performance for the first few models as a sample
model_performance[:3]  # Show performance for the first 3 years as an example

