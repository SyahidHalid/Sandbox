# chatgpt : example Principal Component Analysis (PCA) when work as a data scientist in bank using pandas 
#Here's an example of how a data scientist in a bank might use Principal Component Analysis (PCA) to reduce dimensionality while analyzing customer transaction data. We'll use pandas for data manipulation and sklearn for PCA.

# Use Case: Customer Credit Card Transactions
# Scenario:
# A bank wants to analyze customer spending patterns across multiple categories (e.g., groceries, travel, entertainment, etc.). 
# The goal is to reduce the number of features while retaining maximum variance in the data, making it easier for clustering or classification tasks.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample data: Customer spending across different categories
data = {
    'Customer_ID': [101, 102, 103, 104, 105],
    'Groceries': [200, 150, 180, 220, 130],
    'Travel': [500, 700, 650, 620, 720],
    'Dining': [300, 350, 320, 330, 310],
    'Entertainment': [100, 200, 150, 180, 120],
    'Shopping': [400, 500, 450, 470, 480]
}

# Create DataFrame
df = pd.DataFrame(data)

# Remove Customer_ID before applying PCA
X = df.drop(columns=['Customer_ID'])

# Standardize the data (PCA is sensitive to scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with principal components
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Customer_ID'] = df['Customer_ID']

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Display results
print("Explained Variance Ratio:", explained_variance)
print("\nTransformed Data (First Two Principal Components):")
print(pca_df)

# Plot the principal components
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c='blue', edgecolors='k')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Result - Customer Segmentation")
plt.grid()
plt.show()

#Explanation of the PCA Output (Scatter Plot)
#Each Point Represents a Customer:

#The scatter plot shows customers projected into a two-dimensional PCA space.
#Each dot corresponds to a customer’s transaction pattern transformed into Principal Component 1 (PC1) and Principal Component 2 (PC2).
#Principal Components (PCs):

#PC1 (X-axis): The most important component, capturing the highest variance in the original data.
#PC2 (Y-axis): The second most significant component, capturing additional variance.

#Data Spread & Clustering:

#The points are well spread out, indicating that the PCA effectively captured variations in customer spending behavior.
#If we were clustering, we might identify groups of similar spending behaviors.
#Interpretation of Positions:

#Customers far apart in the PCA space have different spending habits.
#Customers closer together have similar spending behaviors.

#Example:
#A point far left on PC1 could represent a customer who spends heavily on one category (e.g., travel).
#A point far right might indicate a customer with a more balanced spending pattern.

#Conclusion:
#Dimensionality Reduction Success: The five original spending categories were compressed into two principal components while retaining important variance.
#Customer Segmentation Potential: This can be used for further analysis, such as clustering (K-Means) or classification (customer profiling).

#Next Steps:
#dentify which original features contribute most to each principal component (using PCA loadings).
#Perform clustering (e.g., K-Means) to group customers into segments based on their spending patterns.
#Use PCA-transformed data for fraud detection, credit risk modeling, or targeted marketing strategies.
