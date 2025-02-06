#-Means, silhouette method, and clustering for dimensionality reduction

#https://medium.com/@volzhinnv/k-means-silhouette-method-and-clustering-for-dimensionality-reduction-cd50e016d36e


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

dataset = make_blobs(n_samples=600, centers=5, cluster_std=[1.1, 0.6, 0.5, 1.2, 1.1], random_state=32)
X = dataset[0]
plt.scatter(X[:, 0], X[:, 1], color = 'grey')
plt.show()

from sklearn.cluster import KMeans

inertia = list()
for i in range(2, 13):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(2, 13), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

from sklearn.metrics import silhouette_score

sil_scores = list()
for i in range(2, 13):
    kmeans = KMeans(n_clusters = i, random_state = 42)
    kmeans.fit(X)
    sil_scores.append(silhouette_score(X, kmeans.labels_))

plt.plot(range(2, 13), sil_scores, color = 'salmon')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silouette metrics')
plt.axvline(x = sil_scores.index(max(sil_scores))+2, linestyle = 'dotted', color = 'red') 
plt.show()

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)



kmeans = KMeans(n_clusters = 50, random_state = 42)

X_cluster_train = kmeans.fit_transform(X_train)
X_cluster_test = kmeans.transform(X_test)

log_reg.fit(X_cluster_train, y_train)
log_reg.score(X_cluster_test, y_test)

