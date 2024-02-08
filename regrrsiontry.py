# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 19:55:16 2023

@author: hatem
"""
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
def remove_outliers(data, feature_names, iqr_multiplier=1.5):
    """
    Remove outliers from specific features using the Interquartile Range (IQR) method.

    Parameters:
        data (DataFrame): The input DataFrame.
        feature_names (list): List of feature names to remove outliers from.
        iqr_multiplier (float): Multiplier to adjust the IQR range for outlier detection.

    Returns:
        DataFrame: DataFrame with outliers removed.
    """
    data_no_outliers = data.copy()
    
    for feature in feature_names:
        Q1 = data_no_outliers[feature].quantile(0.25)
        Q3 = data_no_outliers[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        data_no_outliers = data_no_outliers[
            (data_no_outliers[feature] >= lower_bound) &
            (data_no_outliers[feature] <= upper_bound)
        ]
    
    return data_no_outliers




# Load the dataset
insurance = pd.read_csv("insurance3.csv")

# Check for null values
print(insurance.isnull().sum())
print(insurance.head())



# Data Visualization Before Removing The Ouliers
# Scatter matrix
sns.pairplot(insurance.drop(columns=['sex','region','smoker']),height=1.5, aspect=1.2)

plt.show()

# Correlation matrix
correlation_matrix = insurance.drop(columns=['sex','region','smoker']).corr()
plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation')
plt.show()
# Data Visualization After Removing The Ouliers
features_to_remove_outliers = ['age', 'bmi']
insurance_no_outliers = remove_outliers(insurance, features_to_remove_outliers,2.0)
print("Original dataset shape:", insurance.shape)
print("Dataset shape after removing outliers:", insurance_no_outliers.shape)
print("-----------------------------------------------------------------")
# Scatter matrix
sns.pairplot(insurance_no_outliers.drop(columns=['sex','region','smoker']),height=1.5, aspect=1.2)
plt.show()

# Correlation matrix
correlation_matrix = insurance_no_outliers.drop(columns=['sex','region','smoker']).corr()
plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation after removing outliers')
plt.show()


# Convert categorical data to numerical using one-hot encoding
encoder = OneHotEncoder(sparse=False)
categorical_features = ['region']
encoded_categories = encoder.fit_transform( insurance_no_outliers[categorical_features])
encoded_df = pd.concat([
    insurance_no_outliers.drop(columns=categorical_features),
    pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(categorical_features))
], axis=1)

# Convert other categorical features using label encoding
encoder1 = LabelEncoder()
categorical_features_label = ['sex', 'smoker']
encoded_df[categorical_features_label] = encoded_df[categorical_features_label].apply(encoder1.fit_transform)

# removing null values after encoding
encoded_df=encoded_df.dropna()




# Prepare data for modeling
x = encoded_df.drop('charges', axis=1)
y = encoded_df['charges']
# Get the column/feature names
feature_names = x.columns

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Initialize the Linear Regression model
reg = linear_model.LinearRegression()

# Backward Elimination
n_features_to_select = x_train.shape[1]  # Start with all features

for i in range(n_features_to_select, 0, -1):
    rfe = RFE(estimator=reg, n_features_to_select=i)
    x_train_selected = rfe.fit_transform(x_train_scaled, y_train)
    x_test_selected = rfe.transform(x_test_scaled)

    # Get the indices of selected features
    selected_indices = np.where(rfe.support_)[0]
    selected_features = feature_names[selected_indices]
    
    # Train the model
    reg.fit(x_train_selected, y_train)
    y_pred_test = reg.predict(x_test_selected)

    # Calculate MSE
    mse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f'With {i} features:')
    print(f'Selected features: {selected_features}')
    print(f'Test score (R2): {reg.score(x_test_selected, y_test)}')
    print(f'MSE on test: {mse_test}')
    print("-----------------------------------------------------------------")