# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 17:26:30 2025

@author: MD ANISH
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Load the house prediction dataset
df =pd.read_csv(r'D:\HOUSE_PRED.csv')

# Drop the first row which contains the table name
df = df.iloc[1:]

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Check for any missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Select features for clustering (excluding price if it's the target variable)
features = ['square_feet', 'num_rooms', 'age', 'distance_to_city(km)']
X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Create elbow plot
elbow_df = pd.DataFrame({'Clusters': range(1, 11), 'WCSS': wcss})
fig_elbow = px.line(elbow_df, x='Clusters', y='WCSS', 
                   title='Elbow Method for Optimal Number of Clusters',
                   markers=True)
fig_elbow.show()

# Apply K-Means with optimal number of clusters (let's choose 3 based on elbow)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataframe
df['Cluster'] = clusters

# Calculate cluster statistics
cluster_stats = df.groupby('Cluster').agg({
    'square_feet': ['mean', 'std'],
    'num_rooms': ['mean', 'std'],
    'age': ['mean', 'std'],
    'distance_to_city(km)': ['mean', 'std'],
    'price': ['mean', 'std']
}).round(2)

# Create scatter plot to visualize clusters
fig_clusters = px.scatter(df, x='square_feet', y='price', 
                         color='Cluster',
                         title='House Clusters by Square Feet and Price',
                         labels={'square_feet': 'Square Feet', 'price': 'Price'},
                         hover_data=['num_rooms', 'age', 'distance_to_city(km)'])

fig_clusters.update_layout(plot_bgcolor='white')
fig_clusters.show()

# Return cluster statistics and the clustered dataframe
print("Cluster Statistics:")
cluster_stats