# As a data scientist, using K-Means clustering can help segment data based on similarities, even 
# when you don’t have labeled data. Here’s an example of how you might apply K-Means clustering 
# in a real-world scenario, such as customer segmentation based on purchasing behavior in a retail 
# dataset.
# chatgpt : example k-means when work as a data scientist

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Load the dataset (for example, customer purchase data)
# Example dataset structure: CustomerID, Age, Annual_Income, Spending_Score
#data = pd.read_csv("customer_data.csv")

np.random.seed(42)
num_customers = 200

data = {
    "CustomerID": range(1, num_customers + 1),
    "Age": np.random.randint(18, 70, num_customers),
    "Annual_Income": np.random.randint(15000, 100000, num_customers),
    "Spending_Score": np.random.randint(1, 100, num_customers),
}

# Create a DataFrame
data = pd.DataFrame(data)

# Display the first few rows of the dataset
print(data.head())


# Select the features for clustering (e.g., Annual Income and Spending Score)
X = data[['Annual_Income', 'Spending_Score']]

# Normalize the features (K-Means performs better on normalized data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
# Let's assume we want to group the customers into 5 segments (k=5)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Get the cluster labels
data['Cluster'] = kmeans.labels_

# Display the centroids of each cluster
centroids = kmeans.cluster_centers_
print("Centroids:\n", centroids)


# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=data['Cluster'], s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='x')
plt.title("Customer Segments Using K-Means")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.show()

# Optional: Analyze each segment
for cluster in range(5):
    print(f"\nCluster {cluster}:")
    print(data[data['Cluster'] == cluster].describe())

#Explanation:
#Dataset: A dataset of customer information with columns such as CustomerID, Age, Annual_Income, and Spending_Score. We focus on clustering based on Annual_Income and Spending_Score.
#Normalization: Features are scaled using StandardScaler because K-Means is sensitive to the scale of data.
#K-Means Clustering: We fit the K-Means model to the data and specify n_clusters=5 (for 5 customer segments).
#Cluster Labels: After fitting the model, we assign the cluster labels back to the original dataset.
#Visualization: A scatter plot shows the customer segments, with cluster centroids marked in red.
#Use Cases:
#Customer Segmentation: Group customers by purchasing patterns or demographics.
#Product Segmentation: Group products based on features like price, demand, and category.
#Behavioral Analytics: Segment users on websites or apps based on actions taken (e.g., click behavior).
#K-Means clustering helps in exploratory data analysis (EDA) and can provide actionable insights for personalized marketing, inventory management, or even pricing strategies.

# snip graph
# chatgpt : please help explain and conclude the output of k-means clustering


#The scatter plot represents the results of K-Means clustering applied to customer segmentation based on Annual Income and Spending Score (both scaled for standardization).

#Explanation of the Plot:
#Data Points (Small Colored Dots): Each dot represents a customer, plotted based on their annual income (x-axis) and spending score (y-axis).
#Clusters (Different Colors): Customers are grouped into distinct clusters, each represented by a different color.
#Centroids (Red X Marks): The red X marks indicate the center of each cluster, which is the mean position of all customers in that cluster.
#Scaled Axes: The values are standardized (mean = 0, standard deviation = 1) to ensure fair distance measurement.

#Key Observations & Conclusion:
#The model identified five distinct customer segments, suggesting five different types of shopping behaviors.
#Customers with high spending scores and high incomes (top right, yellow) may represent premium shoppers.
#Customers with low spending scores and low incomes (bottom left, green) may be budget-conscious shoppers.
#Some clusters have more spread-out customers, indicating variation in spending behavior within those groups.
#The centroids (red X) indicate the "average customer" for each segment, helping businesses target specific groups for personalized marketing.

#Business Implications:
#High-income, high-spending customers can be targeted for luxury products.
#Low-income, low-spending customers may respond better to discount offers.
#Mid-range spenders could be encouraged to increase spending via promotions.
#In summary, K-Means clustering effectively segments customers based on spending patterns, allowing businesses to tailor their strategies for different customer groups. 🚀

#=====================================================================================================
# Buat dataset
# chatgpt : create me a data set example of customer purchase data using pandas
from datetime import datetime, timedelta

# Number of records
n = 100

# Sample data
np.random.seed(42)

# Generate Customer IDs
customer_ids = np.random.randint(1000, 1100, size=n)

# Generate product names
products = ['Laptop', 'Smartphone', 'Headphones', 'Monitor', 'Keyboard', 'Mouse', 'Tablet']
product_choices = np.random.choice(products, size=n)

# Generate quantities
quantities = np.random.randint(1, 5, size=n)

# Generate prices (based on product)
prices = {
    'Laptop': 1200,
    'Smartphone': 800,
    'Headphones': 150,
    'Monitor': 300,
    'Keyboard': 50,
    'Mouse': 30,
    'Tablet': 500
}
price_list = [prices[product] for product in product_choices]

# Generate purchase dates
start_date = datetime(2024, 1, 1)
date_range = [start_date + timedelta(days=int(np.random.randint(0, 365))) for _ in range(n)]

# Create DataFrame
data = {
    'CustomerID': customer_ids,
    'Product': product_choices,
    'Quantity': quantities,
    'Price': price_list,
    'TotalPrice': quantities * np.array(price_list),
    'PurchaseDate': date_range
}
