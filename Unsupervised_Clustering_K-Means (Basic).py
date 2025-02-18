# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
# X is the data points, y is the true labels (not used for clustering)
# make_blobs is used to generate synthetic 2-dimensional data
X, y = make_blobs(n_samples=300,
                  centers=4,
                  cluster_std=0.60,
                  random_state=0)


# Plot the data points
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Data points")
plt.show()

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4)  # Assuming 4 clusters
kmeans.fit(X)

# Get the cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the clustered data
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

# Plot the centroids of the clusters
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
plt.title("K-Means Clustering")
plt.show()