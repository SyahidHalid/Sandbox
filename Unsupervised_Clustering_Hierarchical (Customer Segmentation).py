# chatgpt : example Hierarchical when work as a data scientist using python

#Example: Hierarchical Clustering for Customer Segmentation

#As a data scientist, you might use Hierarchical Clustering to group customers based on spending behavior, similar to K-Means but without needing to specify the number of clusters beforehand.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

np.random.seed(42)
num_customers = 200

data = {
    "CustomerID": range(1, num_customers + 1),
    "Age": np.random.randint(18, 70, num_customers),
    "Annual_Income": np.random.randint(15000, 100000, num_customers),
    "Spending_Score": np.random.randint(1, 100, num_customers),
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

X = df[['Annual_Income', 'Spending_Score']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform hierarchical clustering and create a dendrogram
linked = linkage(X_scaled, method='ward')  # Ward minimizes variance within clusters

plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer Index")
plt.ylabel("Distance")
plt.show()

# Choose the number of clusters based on dendrogram
num_clusters = 5  # Determined visually from the dendrogram

# Apply Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')
df['Cluster'] = hc.fit_predict(X_scaled)

# Visualize the Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['Cluster'], palette='viridis')
plt.title("Customer Segmentation Using Hierarchical Clustering")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.legend(title="Cluster")
plt.show()

# Analyze each segment
for cluster in range(num_clusters):
    print(f"\nCluster {cluster}:")
    print(df[df['Cluster'] == cluster].describe())


# Explanation
#Feature Selection: We use Annual_Income and Spending_Score for segmentation.
#Standardization: Standardizing ensures distance-based clustering works properly.
#Dendrogram Analysis: We visualize hierarchical relationships between customers.
#Cluster Assignment: We set n_clusters=5 based on the dendrogram.
#Visualization: Scatter plot shows how customers are grouped.
#Advantages of Hierarchical Clustering
#✅ No need to predefine the number of clusters (can be determined visually).
#✅ Works well for small to medium datasets.
#✅ Provides hierarchical relationships between customers.

#Use Cases
#Customer Segmentation (e.g., high spenders vs. budget shoppers)
#Anomaly Detection (e.g., identifying unusual spending behaviors)
#Genetic Data Clustering (e.g., grouping similar DNA sequences)


# chatgpt : please help explain and conclude the output of Hierarchical clustering

#Explanation of the Hierarchical Clustering Output
#The scatter plot represents customer segmentation using Hierarchical Clustering based on Annual Income and Spending Score (both scaled).

#Key Observations
#Clusters (Different Colors):

#The model has grouped customers into 5 distinct clusters (0, 1, 2, 3, 4).
#Each color represents a different customer segment with unique spending behavior.
#Scaled Axes:

#X-axis (Annual Income - scaled): Indicates how much a customer earns annually.
#Y-axis (Spending Score - scaled): Represents spending behavior (low vs. high spenders).
#Cluster Analysis:

#Cluster 0 (Purple): Customers with moderate income but diverse spending habits.
#Cluster 1 (Dark Blue): Lower-income customers with low to moderate spending.
#Cluster 2 (Cyan): High-income customers with moderate spending patterns.
#Cluster 3 (Green): Low-income customers with low spending scores.
#Cluster 4 (Yellow): High spenders with high income, possibly VIP customers.

#Comparison with K-Means
#Aspect K-Means Clustering	Hierarchical Clustering
#Cluster  Assignment	Predefined number (k)	Based on dendrogram
#Flexibility	Less flexible	More flexible
#Interpretability	No hierarchy	Provides hierarchical structure
#Scalability	Works well for large data	Can be computationally expensive


#Conclusion

#Hierarchical clustering effectively segments customers without predefining the number of clusters.
#It helps businesses identify spending patterns for targeted marketing strategies.
#The dendrogram helps visualize relationships between customer groups.


#💡 Business Application:

#Luxury segment (Cluster 4 - Yellow): High-income, high-spending customers—ideal for premium services.
#Budget-conscious shoppers (Cluster 3 - Green): Low-income, low-spending customers—best targeted with discounts.
#Mid-range spenders (Clusters 1 & 2): Moderate spenders who can be encouraged with promotions.
